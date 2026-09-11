# -*- coding: utf-8 -*-
"""Canonical empirical-subspace operators for TC-MaOEA.

Global covariance directions are not local manifold tangents. The fitted EEJ is
normalized-objective ridge regression, not an exact derivative. Orthogonality
below refers only to the computed linear subspaces, before boundary repair.
"""
from __future__ import annotations

from typing import Tuple, Optional, Union, Any
import numpy as np

from core.population import Population
from util.array_backend import to_numpy
from util.nds.non_dominated_sorting import NonDominatedSorting
try:
    from core.nds.gpu_nds import boolean_matrix_nds as _nds_fast
except Exception:
    _nds_fast = None


def population_matrix(pop: Population, key: str) -> np.ndarray:
    """Transfer individual arrays before stacking (including device tensors)."""
    return np.asarray([to_numpy(row) for row in pop.get(key, to_numpy=False)], dtype=float)


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(to_numpy(value), dtype=float)
    if array.ndim != 2 or array.shape[1] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix with nonzero column count")
    return array


def project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {w >= 0: sum(w) = z} (Duchi et al.)."""
    v = np.asarray(to_numpy(v), dtype=float)
    if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("v must be a nonempty finite vector")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    # Translation invariance avoids cancellation for a large common offset.
    shifted = v - np.max(v)
    u = np.sort(shifted)[::-1]
    cssv = np.cumsum(u) - z
    rho = np.flatnonzero(u > cssv / np.arange(1, len(v) + 1))[-1]
    return np.maximum(shifted - cssv[rho] / (rho + 1), 0.0)


def certify_empirical_descent(J: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return direction only if every fitted objective strictly decreases.

    A scale-aware roundoff margin excludes unresolved signs. Zero is the safe
    no-step result, not a certificate of stationarity of the true objectives.
    Projection or boundary reflection requires a new certificate; neither
    inherits the certificate of the unprojected direction.
    """
    J = _matrix(J, "J")
    d = np.asarray(to_numpy(direction), dtype=float)
    if d.shape != (J.shape[1],) or not np.all(np.isfinite(d)):
        raise ValueError("direction must be finite and match J columns")
    norm_d = float(np.linalg.norm(d))
    if norm_d <= 1e-14:
        return np.zeros_like(d)
    margin = 32 * np.finfo(float).eps * np.linalg.norm(J, axis=1) * norm_d
    if not np.all(J @ d < -margin):
        return np.zeros_like(d)
    return d.copy()


def certified_projected_normal_descent(
    J: np.ndarray,
    P_N: np.ndarray,
    d_kkt: np.ndarray,
    delta_de: np.ndarray,
    scale_normal_step: bool = True,
) -> np.ndarray:
    """Compute normal displacement P_N d_kkt with strict descent certification.

    Proposition 2 (TC-MaOEA 2.0): If P_N d_kkt produces objective ascent on any
    fitted objective (J P_N d_kkt >= 0), it is immediately suppressed to zero,
    preventing catastrophic divergence from the manifold.
    """
    J = _matrix(J, "J")
    P_N = _matrix(P_N, "P_N")
    d_kkt = np.asarray(to_numpy(d_kkt), dtype=float)
    pn_dkkt = P_N @ d_kkt
    norm_pn = float(np.linalg.norm(pn_dkkt))
    if norm_pn <= 1e-14:
        return np.zeros_like(pn_dkkt)

    # Check strict descent on fitted Jacobian
    proj_margin = 1e-8 * float(np.linalg.norm(J)) * norm_pn
    if np.any(J @ pn_dkkt >= -proj_margin):
        # Objective ascent detected: strictly suppress to zero
        return np.zeros_like(pn_dkkt)

    mean_de = float(np.mean(np.linalg.norm(delta_de, axis=1))) if delta_de.size else 0.0
    if not scale_normal_step or mean_de <= 1e-14:
        scale = 1.0
    else:
        scale = min(1.0, mean_de / (norm_pn + 1e-12))
    return scale * pn_dkkt


def stochastic_normal_diffusion(
    P_N: np.ndarray,
    n_samples: int,
    xl: np.ndarray,
    xu: np.ndarray,
    sigma_n: float,
    rng: Optional[Union[np.random.Generator, np.random.RandomState]] = None,
) -> np.ndarray:
    """Generate stochastic Gaussian diffusion strictly confined to the normal subspace N(M).

    Drives distance-related variables towards the manifold when J is flat or uncertain.
    """
    P_N = _matrix(P_N, "P_N")
    D = P_N.shape[0]
    xl = np.asarray(to_numpy(xl), dtype=float)
    xu = np.asarray(to_numpy(xu), dtype=float)
    span = np.maximum(xu - xl, 1e-6)

    generator = rng if rng is not None else np.random.default_rng()
    if hasattr(generator, "normal"):
        raw = generator.normal(0.0, 1.0, size=(n_samples, D))
    else:
        raw = generator.randn(n_samples, D)

    # Project noise into normal subspace: xi in N(M)
    proj_noise = raw @ P_N.T
    return float(sigma_n) * span[None, :] * proj_noise


def solve_kkt_simplex_qp(
    J: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximately minimize ||J.T lambda||^2 over the unit simplex.

    Projected gradient uses a simplex dual-gap stopping test on the scaled
    Gram matrix. Iteration exhaustion does not establish optimality. Returned
    weights remain feasible; the returned direction is -J.T lambda only when
    it independently passes strict empirical common-descent certification,
    otherwise zero. No true-function or nonlinear finite-step guarantee follows.
    """
    J = _matrix(J, "J")
    M, D = J.shape
    if M == 0 or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("J needs rows and max_iter/tol must be positive")
    lam = np.full(M, 1.0 / M)
    scale = float(np.max(np.abs(J)))
    if scale == 0:
        return lam, np.zeros(D)
    A = J / scale
    G = A @ A.T
    lipschitz = float(np.linalg.eigvalsh(G)[-1])
    for _ in range(max_iter):
        grad = G @ lam
        gap = float(lam @ grad - np.min(grad))
        if gap <= tol * max(1.0, lipschitz):
            break
        lam = project_simplex(lam - grad / lipschitz)
    return lam, certify_empirical_descent(J, -J.T @ lam)


def svd_manifold_decomposition(
    F_norm: np.ndarray,
    tau_var: float = 1e-3,
    tau_gap: float = 50.0,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Return retained covariance dimension r, basis, and covariance spectrum.

    r is empirical, not the PF or PS dimension. Insufficient samples or zero
    covariance return r=0 and an empty basis, never artificial unit eigenvalues.
    """
    F_norm = _matrix(F_norm, "F_norm")
    N, M = F_norm.shape
    if not (0 < tau_var <= 1) or not np.isfinite(tau_gap) or tau_gap <= 1:
        raise ValueError("Require 0 < tau_var <= 1 and finite tau_gap > 1")
    if N < 2 or M < 2:
        return 0, np.empty((M, 0)), np.zeros(M)
    centered = F_norm - np.mean(F_norm, axis=0, keepdims=True)
    covariance = centered.T @ centered / (N - 1)
    _, sigmas, Vt = np.linalg.svd(covariance, full_matrices=False)
    if sigmas[0] == 0:
        return 0, np.empty((M, 0)), sigmas
    k_floor = int(np.count_nonzero(sigmas / sigmas[0] >= tau_var))
    gaps = sigmas[:-1] / np.maximum(sigmas[1:], np.finfo(float).tiny)
    candidates = np.flatnonzero(gaps >= tau_gap)
    k_gap = int(candidates[0]) + 1 if candidates.size else M - 1
    r = min(k_floor, k_gap, M - 1, N - 1)
    return r, Vt[:r].T, sigmas


def compute_eej(
    X_elite: np.ndarray,
    F_norm: np.ndarray,
    reg_scale: float = 1e-6,
) -> np.ndarray:
    """Fit centered normalized-objective ridge regression using a linear solve.

    rho = reg_scale * trace(Xbar.T Xbar) / D. With zero decision spread
    or fewer than two samples the minimum-norm empirical estimate is zero.
    """
    X_elite = _matrix(X_elite, "X_elite")
    F_norm = _matrix(F_norm, "F_norm")
    N, D = X_elite.shape
    M = F_norm.shape[1]
    if len(F_norm) != N or not np.isfinite(reg_scale) or reg_scale <= 0:
        raise ValueError("Aligned samples and positive finite reg_scale required")
    if N < 2:
        return np.zeros((M, D))
    X_bar = X_elite - np.mean(X_elite, axis=0, keepdims=True)
    F_bar = F_norm - np.mean(F_norm, axis=0, keepdims=True)
    gram = X_bar.T @ X_bar
    trace = float(np.trace(gram))
    if trace == 0:
        return np.zeros((M, D))
    rho = max(reg_scale * trace / D, np.finfo(float).tiny)
    system = gram + rho * np.eye(D)
    rhs = X_bar.T @ F_bar
    try:
        return np.linalg.solve(system, rhs).T
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0].T


def build_pullback_projectors(
    J: np.ndarray,
    V_dstar: np.ndarray,
    gamma: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthogonal projectors onto range(B) and its complement.

    B = J.T solve(J J.T + gamma I, V). This is a damped right inverse,
    not Moore-Penrose inversion. SVD retains singular values above
    eps * max(B.shape) * s_max, so rank q may be smaller than r.
    """
    J = _matrix(J, "J")
    V = np.asarray(to_numpy(V_dstar), dtype=float)
    M, D = J.shape
    if V.ndim != 2 or V.shape[0] != M or not np.all(np.isfinite(V)):
        raise ValueError("V must be finite with one row per objective")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if V.shape[1] == 0:
        return np.zeros((D, D)), np.eye(D)
    B = J.T @ np.linalg.solve(J @ J.T + gamma * np.eye(M), V)
    U, singular, _ = np.linalg.svd(B, full_matrices=False)
    cutoff = np.finfo(float).eps * max(B.shape) * singular[0]
    Q = U[:, singular > cutoff]
    P_T = Q @ Q.T
    return P_T, np.eye(D) - P_T


def reflective_clamp(y: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    """Repeated reflection into finite box bounds, including fixed coordinates."""
    y = np.asarray(to_numpy(y), dtype=float)
    xl = np.asarray(to_numpy(xl), dtype=float)
    xu = np.asarray(to_numpy(xu), dtype=float)
    if y.ndim not in (1, 2) or not all(np.all(np.isfinite(a)) for a in (y, xl, xu)):
        raise ValueError("Finite vectors or batches and finite bounds required")
    lo, hi = np.broadcast_arrays(xl, xu)
    if lo.shape != (y.shape[-1],) or np.any(hi < lo):
        raise ValueError("Bounds must match the decision dimension and xl <= xu")
    width = hi - lo
    safe_width = np.where(width > 0, width, 1.0)
    phase = np.remainder(y - lo, 2.0 * safe_width)
    reflected = lo + safe_width - np.abs(phase - safe_width)
    return np.where(width > 0, reflected, lo)


def apd_environmental_selection(
    pool: Population,
    W_adapt: np.ndarray,
    z_min: np.ndarray,
    z_max: np.ndarray,
    n_survive: int,
    t_ratio: float,
    alpha: float = 2.0,
) -> Population:
    """Nondominated fronts followed by occupancy-aware APD critical-front filling.

    Complete fronts have priority. In the split front, repeatedly choose a
    least-occupied available niche, then its minimum-APD candidate. Ties use
    APD then index, making selection reproducible without global RNG state.
    """
    if n_survive < 0:
        raise ValueError("n_survive must be nonnegative")
    if len(pool) <= n_survive:
        return pool
    if n_survive == 0:
        return pool[:0]
    F = _matrix(population_matrix(pool, "F"), "F")
    W = _matrix(W_adapt, "W_adapt")
    if len(W) == 0 or W.shape[1] != F.shape[1]:
        raise ValueError("At least one matching reference direction required")
    norms_w = np.linalg.norm(W, axis=1)
    if np.any(norms_w == 0):
        raise ValueError("Reference directions must be nonzero")
    W = W / norms_w[:, None]
    span = np.maximum(np.asarray(z_max) - np.asarray(z_min), 1e-12)
    F_trans = (F - z_min) / span
    norm_f = np.linalg.norm(F_trans, axis=1)
    theta = np.arccos(np.clip(F_trans @ W.T / np.maximum(norm_f[:, None], 1e-12), -1, 1))
    assoc = np.argmin(theta, axis=1)
    angles = np.arccos(np.clip(W @ W.T, -1, 1))
    np.fill_diagonal(angles, np.inf)
    gamma = np.maximum(np.min(angles, axis=1), 1e-6) if len(W) > 1 else np.array([np.pi])
    scores = norm_f * (1 + F.shape[1] * np.clip(t_ratio, 0, 1) ** alpha * theta[np.arange(len(F)), assoc] / gamma[assoc])
    survivors = []
    fronts = _nds_fast(F) if _nds_fast is not None else NonDominatedSorting().do(F)
    for front in fronts:
        if len(survivors) + len(front) <= n_survive:
            survivors.extend(front)
            continue
        counts = np.bincount(assoc[survivors], minlength=len(W))
        queues = {}
        for idx in sorted(front, key=lambda i: (scores[i], i)):
            queues.setdefault(int(assoc[idx]), []).append(int(idx))
        while len(survivors) < n_survive:
            niche = min(queues, key=lambda k: (counts[k], scores[queues[k][0]], k))
            survivors.append(queues[niche].pop(0))
            counts[niche] += 1
            if not queues[niche]:
                del queues[niche]
        break
    return pool[np.asarray(survivors, dtype=int)]


def epr_environmental_selection(
    pool: Population,
    W_adapt: np.ndarray,
    z_min: np.ndarray,
    z_max: np.ndarray,
    n_survive: int,
    t_ratio: float,
    alpha: float = 2.0,
) -> Population:
    """Extreme-Preserving Reference-Vector Environmental Selection (EPR-Selection).

    Theoretical Advancements over standard APD:
    1. Unconditional Extreme Retention: Vértices extremos da frente são identificados
       via Achievement Scalarizing Functions (ASF) e protegidos incondicionalmente,
       ancorando as fronteiras do hipervolume e eliminando colapso de borda.
    2. Curvature-Invariant Score: Decomposição em d_parallel e d_perp normalizada por
       hiperplano elimina a penalidade radial ||F|| de frentes convexas (ex: ZDT1).
    3. Niche-Occupancy Balancing: Alocação gulosa prioriza nichos com menor contagem,
       impedindo aglomerações e garantindo espalhamento uniforme.
    """
    if n_survive < 0:
        raise ValueError("n_survive must be nonnegative")
    if len(pool) <= n_survive:
        return pool
    if n_survive == 0:
        return pool[:0]

    F = _matrix(population_matrix(pool, "F"), "F")
    N, M = F.shape
    W = _matrix(W_adapt, "W_adapt")
    if len(W) == 0 or W.shape[1] != M:
        raise ValueError("At least one matching reference direction required")

    norms_w = np.linalg.norm(W, axis=1)
    if np.any(norms_w == 0):
        raise ValueError("Reference directions must be nonzero")
    W = W / norms_w[:, None]

    z_min = np.asarray(z_min, dtype=float)
    z_max = np.asarray(z_max, dtype=float)

    # 1. Non-dominated sorting and Unconditional Extreme Point Retention via ASF on Front 0
    fronts = _nds_fast(F) if _nds_fast is not None else NonDominatedSorting().do(F)
    f0 = fronts[0]
    extreme_indices: list[int] = []
    for j in range(M):
        w_asf = np.full(M, 1e-6)
        w_asf[j] = 1.0
        diff = np.maximum(F[f0] - z_min[None, :], 0.0)
        asf_values = np.max(diff / w_asf[None, :], axis=1)
        best_f0_idx = int(np.argmin(asf_values))
        best_idx = int(f0[best_f0_idx])
        if best_idx not in extreme_indices:
            extreme_indices.append(best_idx)

    # 2. Intercept Hyperplane Estimation for True Nadir Scaling
    span = np.maximum(z_max - z_min, 1e-12)
    if len(extreme_indices) == M:
        try:
            E = F[extreme_indices] - z_min[None, :]
            # Solve E @ a = 1
            a = np.linalg.solve(E, np.ones(M))
            if np.all(a > 1e-6) and np.all(np.isfinite(a)):
                span = np.maximum(1.0 / a, 1e-6)
        except Exception:
            pass

    F_norm = np.maximum(F - z_min[None, :], 0.0) / span[None, :]
    norm_f = np.linalg.norm(F_norm, axis=1)

    # 3. Reference Direction Association
    inner_prod = F_norm @ W.T
    cos_theta = np.clip(inner_prod / np.maximum(norm_f[:, None], 1e-12), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    assoc = np.argmin(theta, axis=1)

    # Reference vector neighborhood angle
    angles = np.arccos(np.clip(W @ W.T, -1.0, 1.0))
    np.fill_diagonal(angles, np.inf)
    gamma = np.maximum(np.min(angles, axis=1), 1e-6) if len(W) > 1 else np.array([np.pi])

    # 4. Curvature-Invariant Niche Metric (Psi)
    # d_parallel along associated reference vector
    assoc_w = W[assoc]
    d_parallel = np.sum(F_norm * assoc_w, axis=1)
    perp_vec = F_norm - d_parallel[:, None] * assoc_w
    d_perp = np.linalg.norm(perp_vec, axis=1)

    tau_ratio = float(np.clip(t_ratio, 0.0, 1.0))
    scores = d_parallel + float(M) * (tau_ratio ** alpha) * (d_perp / gamma[assoc])

    survivors: list[int] = []
    # Retain extreme vertices unconditionally
    for e_idx in extreme_indices:
        if len(survivors) < n_survive:
            survivors.append(e_idx)

    for front in fronts:
        # Candidates in this front not already added
        unselected = [idx for idx in front if idx not in survivors]
        if len(survivors) + len(unselected) <= n_survive:
            survivors.extend(unselected)
            if len(survivors) == n_survive:
                break
            continue

        # Split critical front with empty/least-occupied niche priority
        counts = np.bincount(assoc[survivors], minlength=len(W)) if survivors else np.zeros(len(W), dtype=int)
        queues: dict[int, list[int]] = {}
        for idx in sorted(unselected, key=lambda i: (scores[i], i)):
            queues.setdefault(int(assoc[idx]), []).append(int(idx))

        while len(survivors) < n_survive and queues:
            # Pick least-occupied niche, breaking ties with best score in niche
            niche = min(queues, key=lambda k: (counts[k], scores[queues[k][0]], k))
            survivors.append(queues[niche].pop(0))
            counts[niche] += 1
            if not queues[niche]:
                del queues[niche]
        break

    return pool[np.asarray(survivors, dtype=int)]


def safe_polynomial_mutation(
    X: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta_m: float = 20.0,
    prob_m: Optional[float] = None,
    rng: Optional[Union[np.random.Generator, np.random.RandomState]] = None,
) -> np.ndarray:
    """Vectorized polynomial mutation with reflective bound handling."""
    X = np.asarray(X, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    N, D = X.shape
    if prob_m is None:
        prob_m = 1.0 / max(D, 1)

    if rng is None:
        rng = np.random.default_rng()

    if hasattr(rng, "random"):
        rand_m = rng.random((N, D))
        u = rng.random((N, D))
    elif hasattr(rng, "uniform"):
        rand_m = rng.uniform(0.0, 1.0, size=(N, D))
        u = rng.uniform(0.0, 1.0, size=(N, D))
    else:
        rand_m = np.random.random((N, D))
        u = np.random.random((N, D))

    mutate_mask = rand_m < prob_m
    if not np.any(mutate_mask):
        return X.copy()

    X_mut = X.copy()
    diff = xu - xl
    safe_diff = np.where(diff > 0, diff, 1.0)

    delta1 = np.clip((X - xl) / safe_diff, 0.0, 1.0)
    delta2 = np.clip((xu - X) / safe_diff, 0.0, 1.0)

    mut_pow = 1.0 / (eta_m + 1.0)

    xy_left = 1.0 - delta1
    val_left = 2.0 * u + (1.0 - 2.0 * u) * (np.power(np.maximum(xy_left, 0.0), eta_m + 1.0))
    delta_q_left = np.power(np.maximum(val_left, 0.0), mut_pow) - 1.0

    xy_right = 1.0 - delta2
    val_right = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (np.power(np.maximum(xy_right, 0.0), eta_m + 1.0))
    delta_q_right = 1.0 - np.power(np.maximum(val_right, 0.0), mut_pow)

    delta_q = np.where(u <= 0.5, delta_q_left, delta_q_right)
    mutated = X + delta_q * safe_diff
    clipped = np.clip(mutated, xl, xu)
    X_mut[mutate_mask] = clipped[mutate_mask]
    return X_mut
