# -*- coding: utf-8 -*-
# emopylab 2026
"""Tangent-Bundle Pullback Operators for TC-MaOEA.

Implements the mathematical primitives defined in SPEC_TC_MAOEA.md:
1. SVD objective-space manifold estimation and tangent basis V_{d*}.
2. Ensemble Evolutionary Jacobian (EEJ) estimation via Ridge linear regression.
3. Decision-space pullback tangent projector P_T and normal complement P_N.
4. Dual simplex Karush-Kuhn-Tucker (KKT) Pareto descent direction solver.
5. Unified Tangent-Bundle Pullback Variation (TBPV) operator.
"""
from __future__ import annotations

import numpy as np


def compute_manifold_tangent_basis(
    F: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
    tau: float = 0.95,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Estimate intrinsic dimension d* and tangent basis V_{d*} via SVD of objective covariance.

    Parameters
    ----------
    F : np.ndarray
        Objective matrix of elite non-dominated solutions, shape (N_elite, M).
    z_min : np.ndarray
        Ideal point, shape (M,).
    z_nad : np.ndarray
        Nadir point, shape (M,).
    tau : float, default=0.95
        Cumulative energy threshold for intrinsic dimensionality detection.

    Returns
    -------
    d_star : int
        Estimated intrinsic dimension, guaranteed 1 <= d_star <= M - 1.
    V_d : np.ndarray
        Orthonormal tangent bundle basis, shape (M, d_star).
    sigma : np.ndarray
        Ordered singular values of the covariance matrix, shape (M,).
    """
    F_arr = np.asarray(F, dtype=float)
    N, M = F_arr.shape
    if N < 2 or M < 2:
        d_fallback = max(1, M - 1)
        return d_fallback, np.eye(M, d_fallback, dtype=float), np.ones(M, dtype=float)

    span = np.maximum(z_nad - z_min, 1e-12)
    Fn = (F_arr - z_min[None, :]) / span[None, :]
    Fn_centered = Fn - np.mean(Fn, axis=0, keepdims=True)

    cov = (Fn_centered.T @ Fn_centered) / max(N - 1, 1)
    try:
        _, sigma, Vt = np.linalg.svd(cov, full_matrices=False)
        V = Vt.T
    except np.linalg.LinAlgError:
        d_fallback = max(1, M - 1)
        return d_fallback, np.eye(M, d_fallback, dtype=float), np.ones(M, dtype=float)

    total_energy = float(np.sum(sigma**2))
    if total_energy <= 1e-16:
        d_fallback = max(1, M - 1)
        return d_fallback, np.eye(M, d_fallback, dtype=float), np.ones(M, dtype=float)

    # Dual-Criteria Intrinsic Dimension Detection (Spectral Gap + Relative Variance Floor)
    rel_sig = sigma / max(float(sigma[0]), 1e-12)
    significant = np.where(rel_sig >= 1e-3)[0]
    k_sig = int(significant[-1]) + 1 if len(significant) > 0 else 1

    gaps = np.array([sigma[k] / max(float(sigma[k + 1]), 1e-30) for k in range(M - 1)])
    gap_candidates = np.where(gaps >= 50.0)[0]
    if len(gap_candidates) > 0:
        k_gap = int(gap_candidates[0]) + 1
    else:
        cum_energy = np.cumsum(sigma**2) / total_energy
        indices = np.where(cum_energy >= tau)[0]
        k_gap = int(indices[0]) + 1 if len(indices) > 0 else M - 1

    d_star = int(min(k_sig, k_gap))
    # Bound d_star strictly to a sub-manifold
    d_star = int(max(1, min(d_star, M - 1)))
    V_d = V[:, :d_star]
    return d_star, V_d, sigma


def compute_eej_jacobian(
    X: np.ndarray,
    F: np.ndarray,
    reg: float = 1e-6,
) -> np.ndarray:
    """Estimate empirical multi-objective Jacobian J in R^{M x D} via regularized Ridge regression.

    Parameters
    ----------
    X : np.ndarray
        Decision coordinates of elite population, shape (N, D).
    F : np.ndarray
        Objective values of elite population, shape (N, M).
    reg : float, default=1e-6
        Ridge Tikhonov regularization scale factor.

    Returns
    -------
    J : np.ndarray
        Estimated Jacobian matrix, shape (M, D).
    """
    X_arr = np.asarray(X, dtype=float)
    F_arr = np.asarray(F, dtype=float)
    N, D = X_arr.shape
    M = F_arr.shape[1]

    if N < 2:
        return np.zeros((M, D), dtype=float)

    dX = X_arr - np.mean(X_arr, axis=0, keepdims=True)
    dF = F_arr - np.mean(F_arr, axis=0, keepdims=True)

    C_X = dX.T @ dX
    diag_mean = float(np.trace(C_X)) / max(D, 1)
    rho = max(reg * diag_mean, 1e-10)

    reg_mat = C_X + rho * np.eye(D, dtype=float)
    try:
        JT = np.linalg.solve(reg_mat, dX.T @ dF)
        J = JT.T
    except np.linalg.LinAlgError:
        JT, _, _, _ = np.linalg.lstsq(reg_mat, dX.T @ dF, rcond=None)
        J = JT.T

    return J


def build_pullback_projectors(
    J: np.ndarray,
    V_d: np.ndarray,
    gamma: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct orthogonal tangent projector P_T and normal projector P_N in R^{D x D}.

    Uses thin QR decomposition on the pullback basis B_T = J^dagger V_{d*}
    to guarantee exact linear algebraic idempotence (P_T^2 = P_T), symmetry (P_T = P_T^T),
    and orthogonality (P_T P_N = 0) to machine precision.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix, shape (M, D).
    V_d : np.ndarray
        Objective tangent bundle orthonormal basis, shape (M, d_star).
    gamma : float, default=1e-8
        Regularization parameter for Moore-Penrose pseudo-inverse.

    Returns
    -------
    P_T : np.ndarray
        Exact orthogonal tangent projector, shape (D, D), rank d_star.
    P_N : np.ndarray
        Exact orthogonal normal projector, shape (D, D), rank D - d_star.
    """
    J_arr = np.asarray(J, dtype=float)
    V_arr = np.asarray(V_d, dtype=float)
    M, D = J_arr.shape
    d_star = V_arr.shape[1] if V_arr.ndim == 2 else 1

    I_D = np.eye(D, dtype=float)
    if d_star == 0 or np.all(np.abs(J_arr) < 1e-14):
        return np.zeros((D, D), dtype=float), I_D

    # J^dagger = J^T (J J^T + gamma I_M)^{-1} in R^{D x M}
    JJT = J_arr @ J_arr.T
    inv_JJT = np.linalg.pinv(JJT + gamma * np.eye(M, dtype=float))
    J_pinv = J_arr.T @ inv_JJT  # (D, M)

    # Pullback tangent basis in decision space: B_T = J^dagger V_d in R^{D x d*}
    B_T = J_pinv @ V_arr

    # Thin QR factorization yields orthonormal column basis Q_T in R^{D x d*}
    Q_T, _ = np.linalg.qr(B_T)

    # Exact orthogonal projection matrix onto col(B_T)
    P_T = Q_T @ Q_T.T
    P_N = I_D - P_T

    return P_T, P_N


def solve_kkt_descent_simplex(
    J: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> np.ndarray:
    """Solve the dual Karush-Kuhn-Tucker quadratic program on the unit simplex.

    min_{lambda in Delta_{M-1}} || J^T lambda ||_2^2
    s.t. sum_m lambda_m = 1, lambda_m >= 0

    Returns the Pareto descent direction d_KKT = -J^T lambda* in R^D.
    """
    J_arr = np.asarray(J, dtype=float)
    M, D = J_arr.shape
    if M == 1:
        return -J_arr[0].copy()

    H = J_arr @ J_arr.T  # (M, M)
    lam = np.full(M, 1.0 / M, dtype=float)

    # Safe Lipschitz step
    L = float(np.linalg.norm(H, ord=np.inf))
    step = 1.0 / max(L, 1e-12)

    for _ in range(max(max_iter, 1)):
        grad = H @ lam
        # Projected gradient onto unit simplex
        v = lam - step * grad
        lam_next = _project_simplex(v)
        if float(np.linalg.norm(lam_next - lam)) <= tol:
            lam = lam_next
            break
        lam = lam_next

    d_kkt = -J_arr.T @ lam
    return d_kkt


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex sum(x) = 1, x >= 0 (Duchi et al., 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / ind > 0.0
    rho = int(np.where(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def tangent_pullback_variation(
    x_base: np.ndarray,
    delta_de: np.ndarray,
    d_kkt: np.ndarray,
    P_T: np.ndarray,
    P_N: np.ndarray,
    eta_T: float,
    eta_N: float,
    xl: np.ndarray,
    xu: np.ndarray,
) -> np.ndarray:
    """Compute unified Tangent-Bundle Pullback Variation with reflective boundary enforcement."""
    dx = float(eta_T) * (P_T @ delta_de) + float(eta_N) * (P_N @ d_kkt)
    y = x_base + dx

    # Reflective clamping
    below = y < xl
    above = y > xu
    y[below] = 2.0 * xl[below] - y[below]
    y[above] = 2.0 * xu[above] - y[above]
    # Hard safety fallback for extreme overshoots
    return np.clip(y, xl, xu)
