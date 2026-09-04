# emopylab 2026
"""Adaptive Reference-Vector Repair (ARV) module for GCSMaOEA.

This module implements the first core contribution of GCSMaOEA:
**Adaptive Reference-Vector Repair (ARV)**.  The Pareto front of a many-
objective problem is unknown a priori, and fixed reference vectors often
mismatch the true geometry, especially on irregular fronts.  ARV estimates
the current front geometry through an adaptive normal hyperplane and
re-projects the uniform weight vectors onto that hyperplane.  The repaired
vectors then guide the search along the real shape of the front.

Mathematical notation used throughout this module:
    F          : objective matrix, shape (N, M)
    z_min      : ideal point, shape (M,)
    z_nad      : nadir point, shape (M,)
    H          : normal vector of the estimated Pareto-front hyperplane
    W_uniform  : uniformly distributed reference vectors, shape (K, M)
    W_repaired : reference vectors after projection onto H

The normal hyperplane is defined by the implicit equation
    H^T (f - z_min) = 1
so that every projected objective vector f lies on the plane.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Ideal / nadir maintenance
# ---------------------------------------------------------------------------

def update_ideal_nadir(
    F: np.ndarray,
    z_min: Optional[np.ndarray] = None,
    z_nad: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Update the ideal and nadir points from a set of objective vectors.

    This routine is used every generation to keep the reference frame aligned
    with the current non-dominated archive.  The ideal point is the
    component-wise minimum; the nadir point is initialized as the component-
    wise maximum and later refined by ``update_nadir_from_extremes`` using the
    ASF extreme points.

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Objective vectors.
    z_min : np.ndarray, shape (M,) | None
        Previous ideal point.  If None, it is initialized from ``F``.
    z_nad : np.ndarray, shape (M,) | None
        Previous nadir point.  If None, it is initialized from ``F``.

    Returns
    -------
    z_min, z_nad : np.ndarray
        Updated ideal and nadir points.
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        if z_min is not None and z_nad is not None:
            return z_min.copy(), z_nad.copy()
        raise ValueError("Cannot update ideal/nadir from an empty population.")

    f_min = F.min(axis=0)
    f_max = F.max(axis=0)

    if z_min is None:
        z_min = f_min.copy()
    else:
        z_min = np.minimum(z_min, f_min)

    if z_nad is None:
        z_nad = f_max.copy()
    else:
        # The nadir is only moved *outward* here; refinement comes later.
        z_nad = np.maximum(z_nad, f_max)

    return z_min, z_nad


def update_nadir_from_extremes(
    F: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
    extreme_points: np.ndarray,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Refine the nadir point using ASF-detected extreme points.

    Given the extreme points of the current front, this function solves the
    linear system
        (extreme_points - z_min) * alpha = 1_M
    and returns
        z_nad' = 1 / alpha + z_min
    which is a better estimate of the nadir than the raw maximum when the
    front is non-linear or incomplete.

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Objective vectors (fallback if the linear system is singular).
    z_min : np.ndarray, shape (M,)
        Ideal point.
    z_nad : np.ndarray, shape (M,)
        Current nadir point.
    extreme_points : np.ndarray, shape (M, M)
        Matrix whose i-th row is the i-th ASF extreme point.

    Returns
    -------
    z_nad : np.ndarray, shape (M,)
        Refined nadir point.
    """
    M = extreme_points.shape[0]
    z_min = np.asarray(z_min, dtype=float).reshape(-1)
    extreme_points = np.asarray(extreme_points, dtype=float)

    if extreme_points.shape != (M, M):
        return np.asarray(z_nad, dtype=float).reshape(-1)

    z_nad_arr = np.asarray(z_nad, dtype=float).reshape(-1)
    fallback = np.asarray(F, dtype=float).max(axis=0)

    A = extreme_points - z_min
    candidate = fallback
    try:
        if np.linalg.matrix_rank(A) == M:
            alpha = np.linalg.solve(A, np.ones(M, dtype=float))
            inv_alpha = np.full_like(alpha, np.inf, dtype=float)
            np.divide(1.0, alpha, out=inv_alpha, where=alpha > 1e-12)
            refined = inv_alpha + z_min
            # Reject components where the extreme-point solve is unstable; keep
            # the previous nadir there.  This guards badly-scaled fronts where a
            # single extreme blows up one coordinate.
            refined = np.where(
                np.isfinite(refined) & (refined > z_min), refined, z_nad_arr
            )
            candidate = refined
    except Exception:
        candidate = fallback

    # Clamp to stay strictly above the ideal point.
    candidate = np.maximum(candidate, z_min + 1e-12)

    # Optional temporal smoothing (EMA) to avoid nadir oscillation across
    # generations on biased/disparately-scaled objectives.
    beta = float(np.clip(smoothing, 0.0, 1.0))
    if beta > 0.0 and z_nad_arr.shape == candidate.shape and np.all(np.isfinite(z_nad_arr)):
        candidate = beta * z_nad_arr + (1.0 - beta) * candidate
        candidate = np.maximum(candidate, z_min + 1e-12)

    return candidate


# ---------------------------------------------------------------------------
# Extreme-point detection
# ---------------------------------------------------------------------------

def find_extreme_points(
    F: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
) -> np.ndarray:
    """Detect the M ASF extreme points of the current front.

    For each objective axis ``i`` an augmented weight vector is built with a
    large value (1.0) on axis ``i`` and a small value ``epsilon`` elsewhere.
    As specified by Eq. (3) of the article, the ASF value is
        ASF_i(x) = max_m (f_m(x) - z_min_m) / w_m.
    The solution minimizing ASF_i is selected as the i-th extreme point.
    Duplicate rows are retained: on a degenerate front they are evidence that
    the extreme-point system is rank deficient and should trigger the temporal
    hyperplane fallback described in the article.

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Objective vectors.
    z_min : np.ndarray, shape (M,)
        Ideal point.
    z_nad : np.ndarray, shape (M,)
        Nadir point.

    Returns
    -------
    extreme : np.ndarray, shape (M, M)
        Matrix of extreme points (one per row).
    """
    F = np.asarray(F, dtype=float)
    z_min = np.asarray(z_min, dtype=float).reshape(-1)
    # ``z_nad`` remains in the public signature for API compatibility.  The
    # article's ASF definition uses the ideal shift but no ideal/nadir scaling.
    _ = np.asarray(z_nad, dtype=float).reshape(-1)

    _, M = F.shape
    epsilon = 1e-6

    shifted = F - z_min

    extreme = np.zeros((M, M), dtype=float)
    for i in range(M):
        w = np.full(M, epsilon, dtype=float)
        w[i] = 1.0
        asf = np.max(shifted / w, axis=1)
        idx = int(np.argmin(asf))
        extreme[i] = F[idx]

    return extreme


# ---------------------------------------------------------------------------
# Normal hyperplane and vector repair
# ---------------------------------------------------------------------------

def normal_hyperplane(
    extreme_points: np.ndarray,
    z_min: np.ndarray,
    H_prev: Optional[np.ndarray] = None,
    z_nad: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate the normal vector of the Pareto-front hyperplane.

    Given the M ASF extreme points, solve
        (extreme_points - z_min) * H = 1_M
    for the normal vector H.

    On a degenerate front the extreme points become nearly collinear, so the
    linear system is rank-deficient.  In that case we must NOT silently reset
    the geometry to the uniform vector ``1_M`` (that disables the whole
    repair exactly when it is needed most).  Instead we reuse the last valid
    normal vector ``H_prev`` when available (temporal fallback), and only fall
    back to the nadir-based normal ``1 / (z_nad - z_min)`` if no prior estimate
    exists.  This decision is driven by the rank/conditioning of the current
    data, never by the problem identity.

    Parameters
    ----------
    extreme_points : np.ndarray, shape (M, M)
        ASF extreme points, one per row.
    z_min : np.ndarray, shape (M,)
        Ideal point.
    H_prev : np.ndarray, shape (M,) | None
        Last valid normal vector, reused when the current system is singular.
    z_nad : np.ndarray, shape (M,) | None
        Current nadir estimate used by the first-generation fallback.

    Returns
    -------
    H : np.ndarray, shape (M,)
        Normal vector of the estimated front hyperplane.
    """
    M = extreme_points.shape[0]
    A = extreme_points - np.asarray(z_min, dtype=float).reshape(1, -1)

    try:
        if np.linalg.matrix_rank(A) == M:
            H = np.linalg.solve(A, np.ones(M, dtype=float))
            if np.all(np.isfinite(H)) and np.all(H > 1e-12):
                return H
    except Exception:
        pass

    if H_prev is not None:
        H_prev = np.asarray(H_prev, dtype=float).reshape(-1)
        if (
            H_prev.shape == (M,)
            and np.all(np.isfinite(H_prev))
            and np.all(H_prev > 1e-12)
        ):
            return H_prev.copy()

    if z_nad is not None:
        z_nad = np.asarray(z_nad, dtype=float).reshape(-1)
        z_min_arr = np.asarray(z_min, dtype=float).reshape(-1)
        if z_nad.shape == (M,):
            span = z_nad - z_min_arr
            if np.all(np.isfinite(span)) and np.all(span > 1e-12):
                return 1.0 / span

    return np.ones(M, dtype=float)


# ---------------------------------------------------------------------------
# Front-geometry estimation (intrinsic dimension + vector adaptation)
# ---------------------------------------------------------------------------

def estimate_front_dimension(
    F: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
    energy_threshold: float = 0.99,
) -> int:
    """Estimate the intrinsic dimension of the current non-dominated front.

    The normalized objective cloud is centered and its principal-component
    energy spectrum is computed.  The intrinsic dimension is the smallest
    number of components whose cumulative explained variance reaches
    ``energy_threshold``.  This is a purely data-driven probe: a regular
    (M-1)-simplex front returns ``M-1`` while a degenerate curve returns ``1``
    and a 2-D degenerate manifold returns ``2`` -- with no knowledge of which
    benchmark produced the points.

    Returns an integer in ``[1, M]``.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] < 2:
        return int(F.shape[1]) if F.ndim == 2 else 1
    N, M = F.shape

    z_min = np.asarray(z_min, dtype=float).reshape(1, -1)
    z_nad = np.asarray(z_nad, dtype=float).reshape(1, -1)
    denom = np.maximum(z_nad - z_min, 1e-12)
    Fn = (F - z_min) / denom

    centered = Fn - Fn.mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(centered, compute_uv=False)
    except Exception:
        return M - 1 if M > 1 else 1

    energy = sv ** 2
    total = float(np.sum(energy))
    if total <= 1e-18:
        return 1
    cum = np.cumsum(energy) / total
    d_eff = int(np.searchsorted(cum, energy_threshold) + 1)
    return int(np.clip(d_eff, 1, M))


def estimate_dimension_twonn(F: np.ndarray) -> Optional[float]:
    """TwoNN intrinsic-dimension estimate (Facco et al., Sci. Rep. 2017).

    Uses only the ratio ``mu = r2/r1`` of the distances to the two nearest
    neighbors of each point, so it is far more sample-efficient than the
    PCA energy criterion on sparse high-``M`` clouds (the regime where the
    spectral test becomes unreliable, e.g. M=15 with N~200 points).  The
    estimator is the MLE slope ``d = n / sum(log mu_i)`` restricted to the
    smallest 90% of ratios for robustness against outliers.

    Returns ``None`` when there are too few points for a stable estimate.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] < 20:
        return None
    # Pairwise distances (N is population-sized, so O(N^2 M) is acceptable).
    diff = F[:, None, :] - F[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    part = np.partition(dist, 1, axis=1)[:, :2]
    r1 = part[:, 0]
    r2 = part[:, 1]
    valid = (r1 > 1e-12) & np.isfinite(r2)
    if valid.sum() < 10:
        return None
    mu = r2[valid] / r1[valid]
    mu = np.sort(mu)
    n_keep = max(5, int(0.9 * len(mu)))
    mu = mu[:n_keep]
    log_mu = np.log(np.maximum(mu, 1.0 + 1e-12))
    s = float(np.sum(log_mu))
    if s <= 1e-12:
        return None
    return float(len(mu) / s)


def assess_front_degeneracy(
    F: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
    energy_threshold: float = 0.99,
    reduction_min: int = 1,
    gap_min: float = 3.0,
    use_twonn: bool = False,
    twonn_gap_min: float = 3.0,
) -> tuple[int, bool]:
    """Estimate intrinsic dimension and decide whether the front is degenerate.

    The cumulative-energy dimension ``d_eff`` alone is an unreliable degeneracy
    flag on *regular* high-$M$ fronts: once the population converges it clusters
    rather than filling the $(M\\!-\\!1)$-simplex uniformly, so PCA reports a
    rank well below $M-1$ even though the true front is regular.  We therefore
    combine ``d_eff`` with two scale-stable signals measured from the singular
    spectrum $e_1\\ge\\cdots\\ge e_M$ (normalized energies):

      * **dimensional reduction** ``(M-1) - d_eff``: the manifold machinery is
        only worth engaging when it removes a substantial number of dimensions;
        on a near-regular front the reduction is small and simplex repair already
        suffices.  This also matches the coverage bound, whose separation grows
        with $M-1-d^\\star$.
      * **spectral gap** ``e_{d_eff} / e_{d_eff+1}``: a genuinely degenerate
        front shows a sharp cliff after its dominant subspace (e.g.\\ a 1-D curve
        has $e_1\\approx 0.98$, $e_2\\approx 0$), whereas a regular front's energy
        decays gradually.  A gradual decay is rejected even when ``d_eff`` is
        small.

    The front is flagged degenerate iff both
    ``(M-1) - d_eff >= reduction_min`` and the spectral gap ``>= gap_min`` hold.
    On the DTLZ controls this fires on DTLZ5/6 at $M\\ge 8$ (large reduction,
    sharp gap) and stays off on DTLZ1--4 and on low-$M$ cases where simplex
    repair is already adequate.  The decision uses measured data only, never the
    problem identity.

    Returns
    -------
    (d_star, degenerate) : tuple[int, bool]
        Estimated intrinsic dimension and the degeneracy flag.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] < 2:
        M = int(F.shape[1]) if F.ndim == 2 else 1
        return (max(1, M - 1), False)
    N, M = F.shape

    z_min = np.asarray(z_min, dtype=float).reshape(1, -1)
    z_nad = np.asarray(z_nad, dtype=float).reshape(1, -1)
    denom = np.maximum(z_nad - z_min, 1e-12)
    Fn = (F - z_min) / denom
    centered = Fn - Fn.mean(axis=0, keepdims=True)

    try:
        sv = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return (max(1, M - 1), False)

    energy = sv ** 2
    total = float(np.sum(energy))
    if total <= 1e-18:
        return (1, False)
    e = energy / total
    cum = np.cumsum(e)
    d_star = int(np.clip(np.searchsorted(cum, energy_threshold) + 1, 1, M))

    reduction = (M - 1) - d_star
    # Spectral gap between the last in-subspace component and the first excluded.
    if d_star < len(e):
        gap = float(e[d_star - 1] / max(e[d_star], 1e-12))
    else:
        gap = float("inf")

    degenerate = (reduction >= int(reduction_min)) and (gap >= float(gap_min))

    # TwoNN committee member (sample-efficient; rescues sparse high-M clouds
    # where d_eff alone under-reports the reduction).  TwoNN alone never
    # fires: it must be corroborated by the same full spectral gap as the PCA
    # path (a converged *regular* front clusters and fools nearest-neighbor
    # estimators, but its energy decay stays gradual, so the gap test rejects
    # it), and its estimate can only *shrink* d_star, never inflate the
    # degeneracy claim.
    if use_twonn and not degenerate:
        d_tn = estimate_dimension_twonn(centered)
        if d_tn is not None:
            d_tn_round = int(np.clip(round(d_tn), 1, M))
            tn_reduction = (M - 1) - d_tn_round
            if tn_reduction >= int(reduction_min) and gap >= float(twonn_gap_min):
                degenerate = True
                d_star = min(d_star, d_tn_round)

    return (d_star, bool(degenerate))


def generate_manifold_vectors(
    F: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
    d_star: int,
    n_vectors: int,
) -> np.ndarray:
    """Generate reference vectors aligned to the intrinsic front manifold.

    This is the central mechanism that distinguishes GCS-MaOEA from
    repair-based adaptation (RVEA, AR-MOEA, NRV-MOEA).  Those methods keep the
    reference directions on the full ``(M-1)``-simplex and *correct* them after
    the front has already degenerated.  When the true Pareto front has intrinsic
    dimension ``d* < M-1`` (degenerate fronts such as DTLZ5/6, WFG3, MaF6), no
    correction of an ``(M-1)``-dimensional structure can cover a ``d*``-manifold
    without wasting most directions on empty regions.

    Instead, the front manifold itself is estimated by principal-component
    analysis of the current non-dominated objective cloud, and the reference
    vectors are *generated directly* inside the ``d*``-dimensional affine
    subspace ``col(U_{d*})`` where the front lives:

        Fn        = (F - z_min) / (z_nad - z_min)          (normalized cloud)
        mu        = mean(Fn)
        U_{d*}    = top-d* right singular vectors of (Fn - mu)   shape (M, d*)
        T         = (Fn - mu) U_{d*}                       (front in subspace)
        T_grid    = uniform grid over the bounding box of T
        W         = mu + T_grid U_{d*}^T                   (lifted to R^M)

    Because the grid spans the *observed* extent of the front in subspace
    coordinates, every generated vector points at a populated region of the
    manifold.  For a regular ``(M-1)``-front this reduces to a near-uniform
    cover of the simplex, so the operator degrades gracefully and is only
    activated when ``d* < M-1`` (see ``GCSMaOEA._update_reference_frame``).

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Non-dominated objective vectors of the current population.
    z_min, z_nad : np.ndarray, shape (M,)
        Ideal and nadir points used for normalization.
    d_star : int
        Estimated intrinsic dimension of the front (``1 <= d* <= M``).
    n_vectors : int
        Number of reference vectors to generate (the target population size).

    Returns
    -------
    W : np.ndarray, shape (<=n_vectors, M)
        Manifold-aligned reference vectors, max-normalized (largest entry 1).
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] < 2:
        return np.asarray(F, dtype=float).copy()
    N, M = F.shape
    d_star = int(np.clip(d_star, 1, M))
    n_vectors = int(max(1, n_vectors))

    z_min = np.asarray(z_min, dtype=float).reshape(1, -1)
    z_nad = np.asarray(z_nad, dtype=float).reshape(1, -1)
    denom = np.maximum(z_nad - z_min, 1e-12)
    Fn = (F - z_min) / denom

    mu = Fn.mean(axis=0, keepdims=True)
    centered = Fn - mu
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return repair_reference_vectors(Fn, np.ones(M), z_min.reshape(-1))

    U = Vt[:d_star].T                     # (M, d*)
    T = centered @ U                      # (N, d*) front coordinates in subspace

    # Uniform grid over the bounding box of the front in subspace coordinates.
    t_lo = T.min(axis=0)
    t_hi = T.max(axis=0)
    span = np.where(t_hi - t_lo < 1e-12, 1.0, t_hi - t_lo)
    t_lo = t_lo - 0.0 * span              # keep the observed extent
    n_per_dim = int(np.ceil(n_vectors ** (1.0 / d_star)))
    n_per_dim = max(2, n_per_dim)

    axes = [np.linspace(t_lo[j], t_hi[j], n_per_dim) for j in range(d_star)]
    grid = np.stack([g.reshape(-1) for g in np.meshgrid(*axes, indexing="ij")], axis=-1)
    grid = grid.reshape(-1, d_star)

    # Support filter: for d* >= 2 the front need not fill its bounding box
    # (e.g. non-rectangular or partially degenerate patches), so grid nodes
    # farther than one grid spacing from every observed coordinate would point
    # at empty regions.  Keep only supported nodes.
    if d_star >= 2 and len(T) > 0:
        spacing = np.linalg.norm((t_hi - t_lo) / max(n_per_dim - 1, 1))
        d_grid = np.linalg.norm(grid[:, None, :] - T[None, :, :], axis=2).min(axis=1)
        supported = d_grid <= max(spacing, 1e-12)
        if supported.sum() >= 2:
            grid = grid[supported]

    # Trim to exactly n_vectors by farthest-point sampling (preserves spread in
    # any d*, unlike a lexicographic stride over the flattened grid).
    if len(grid) > n_vectors:
        grid = grid[_farthest_point_subset(grid, n_vectors)]

    W = mu + grid @ U.T                   # (K, M) lifted back to objective space
    # Map from normalized space to a conventional weight scale.
    W = np.maximum(W, 1e-12)
    max_vals = W.max(axis=1, keepdims=True)
    max_vals = np.where(max_vals < 1e-12, 1.0, max_vals)
    W = W / max_vals
    # Invariant: always return exactly n_vectors directions.  The support
    # filter can leave fewer grid nodes than requested; top the set up with
    # directions of actual solutions (which lie on the front by construction),
    # chosen farthest-first from the already selected directions.
    if len(W) < n_vectors and len(Fn) > 0:
        W = _top_up_with_solutions(W, Fn, n_vectors)
    return W


def _top_up_with_solutions(W: np.ndarray, Fn: np.ndarray, n_vectors: int) -> np.ndarray:
    """Pad W to n_vectors rows using max-normalized solution directions."""
    cand = np.maximum(np.asarray(Fn, dtype=float), 1e-12)
    cand = cand / cand.max(axis=1, keepdims=True)
    need = int(n_vectors - len(W))
    while need > 0 and len(cand) > 0:
        if len(W) > 0:
            dist = np.linalg.norm(cand[:, None, :] - W[None, :, :], axis=2).min(axis=1)
        else:
            dist = np.full(len(cand), np.inf)
        pick = int(np.argmax(dist))
        W = np.vstack([W, cand[pick][None, :]]) if len(W) else cand[pick][None, :]
        cand = np.delete(cand, pick, axis=0)
        need -= 1
    if need > 0 and len(W) > 0:
        # Not enough distinct solutions: repeat existing directions.
        reps = W[np.arange(need) % len(W)]
        W = np.vstack([W, reps])
    return W


def _farthest_point_subset(points: np.ndarray, k: int) -> np.ndarray:
    """Indices of a k-subset chosen by farthest-point sampling (2-approx cover)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    k = int(min(max(1, k), n))
    chosen = [int(np.argmin(points[:, 0]))]
    mind = np.linalg.norm(points - points[chosen[0]][None, :], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(points - points[nxt][None, :], axis=1))
    return np.asarray(chosen, dtype=int)


def _greedy_cluster_labels(Fn: np.ndarray, radius_factor: float = 2.0) -> np.ndarray:
    """Cluster the normalized cloud by leader clustering with a data-driven radius.

    The radius is ``radius_factor`` times the median nearest-neighbor distance,
    so disconnected components of the front (whose inter-component gaps are
    much larger than the intra-component spacing) fall into separate clusters,
    while a connected front yields a single cluster.  Deterministic and O(N^2).
    """
    n = len(Fn)
    if n < 4:
        return np.zeros(n, dtype=int)
    diff = Fn[:, None, :] - Fn[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    radius = radius_factor * float(np.median(nn[np.isfinite(nn)]))
    if not np.isfinite(radius) or radius <= 0.0:
        return np.zeros(n, dtype=int)
    # Single-linkage components under the threshold radius (BFS).
    adj = dist <= radius
    labels = -np.ones(n, dtype=int)
    comp = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = comp
        while stack:
            u = stack.pop()
            for v in np.where(adj[u] & (labels < 0))[0]:
                labels[v] = comp
                stack.append(int(v))
        comp += 1
    return labels


def generate_manifold_vectors_clustered(
    F: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
    n_vectors: int,
    energy_threshold: float = 0.99,
    max_clusters: int = 6,
    min_cluster_size: int = 5,
) -> np.ndarray:
    """Piecewise (per-cluster) manifold-aligned reference generation.

    A single global PCA basis is only a first-order fit: it fails on
    disconnected fronts (DTLZ7, MaF7, WFG2) and on strongly curved degenerate
    manifolds, where different regions have different local subspaces.  This
    generator clusters the normalized non-dominated cloud with a data-driven
    single-linkage rule, fits a local principal subspace per cluster (each with
    its own local intrinsic dimension), and generates directions inside each
    local subspace, allocating the ``n_vectors`` budget proportionally to
    cluster size.  With one cluster it reduces exactly to the global generator,
    so the piecewise extension can never regress the connected case.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] < 2:
        return np.asarray(F, dtype=float).copy()
    N, M = F.shape
    n_vectors = int(max(1, n_vectors))

    z_min_r = np.asarray(z_min, dtype=float).reshape(1, -1)
    z_nad_r = np.asarray(z_nad, dtype=float).reshape(1, -1)
    denom = np.maximum(z_nad_r - z_min_r, 1e-12)
    Fn = (F - z_min_r) / denom

    labels = _greedy_cluster_labels(Fn)
    unique = [c for c in np.unique(labels) if np.sum(labels == c) >= min_cluster_size]
    if len(unique) <= 1 or len(unique) > int(max_clusters):
        # Connected front (or over-fragmented noise): use the global generator.
        d_star = estimate_front_dimension(F, z_min, z_nad, energy_threshold)
        return generate_manifold_vectors(F, z_min, z_nad, d_star, n_vectors)

    sizes = np.array([np.sum(labels == c) for c in unique], dtype=float)
    budgets = np.maximum(2, np.round(n_vectors * sizes / sizes.sum()).astype(int))

    pieces = []
    for c, budget in zip(unique, budgets):
        Fc = F[labels == c]
        d_c = estimate_front_dimension(Fc, z_min, z_nad, energy_threshold)
        Wc = generate_manifold_vectors(Fc, z_min, z_nad, d_c, int(budget))
        if Wc.ndim == 2 and Wc.shape[1] == M and len(Wc) > 0:
            pieces.append(Wc)
    if not pieces:
        d_star = estimate_front_dimension(F, z_min, z_nad, energy_threshold)
        return generate_manifold_vectors(F, z_min, z_nad, d_star, n_vectors)

    W = np.vstack(pieces)
    if len(W) > n_vectors:
        W = W[_farthest_point_subset(W, n_vectors)]
    elif len(W) < n_vectors:
        # Invariant: exactly n_vectors directions (see generate_manifold_vectors).
        W = _top_up_with_solutions(W, Fn, n_vectors)
    return W


def adapt_reference_vectors(
    W: np.ndarray,
    F: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
    angle_threshold: float = 0.20,
    max_replace_ratio: float = 0.5,
    min_inactive_ratio: float = 0.25,
) -> np.ndarray:
    """Prune inactive reference vectors and resample them onto the front.

    A reference vector is *inactive* when no non-dominated solution lies within
    ``angle_threshold`` (sine distance) of it -- i.e. it points at an empty
    region of objective space, which is exactly what happens on degenerate,
    disconnected or inverted fronts where uniform/simplex vectors over-cover
    the space.  Inactive vectors waste the niching budget.

    The inactive vectors are replaced (in place, keeping ``len(W)`` constant so
    every downstream invariant that depends on the number of directions is
    preserved) by directions taken from actual non-dominated solutions located
    in the *sparsest* covered regions.  Because the replacements are directions
    of real solutions, they automatically lie on the true front manifold,
    however degenerate it is.  The trigger is the measured association count,
    never the problem identity; on a regular front almost every vector is
    active, so the operator is a near no-op and degrades gracefully.

    Parameters
    ----------
    W : np.ndarray, shape (K, M)
        Current reference vectors (e.g. the repaired ones).
    F : np.ndarray, shape (N, M)
        Non-dominated objective vectors.
    z_min, z_nad : np.ndarray, shape (M,)
        Ideal and nadir points.
    angle_threshold : float
        Maximum sine distance for a vector to count as covered.
    max_replace_ratio : float
        Upper bound on the fraction of vectors replaced per call (stability).

    Returns
    -------
    W_new : np.ndarray, shape (K, M)
        Adapted reference vectors (same shape as ``W``).
    """
    W = np.asarray(W, dtype=float)
    F = np.asarray(F, dtype=float)
    if W.ndim != 2 or F.ndim != 2 or len(W) == 0 or len(F) == 0:
        return W.copy() if W.size else W

    K, M = W.shape
    if F.shape[1] != M:
        return W.copy()

    dist = sine_distance(F, W, z_min, z_nad)          # (N, K)
    # A vector is active if at least one solution is within the angular band.
    min_per_vec = dist.min(axis=0)                     # (K,)
    active = min_per_vec <= angle_threshold
    n_active = int(np.sum(active))
    # Act only when a substantial fraction of vectors is inactive, i.e. when the
    # decomposition genuinely over-covers the objective space (degenerate /
    # disconnected / inverted front).  On a regular front almost every vector is
    # active, the inactive fraction is below ``min_inactive_ratio``, and the
    # operator is an exact no-op -- so it cannot regress DTLZ1-4 / WFG4-9.
    n_inactive = K - n_active
    if n_active == K or (n_inactive / float(K)) < float(min_inactive_ratio):
        return W.copy()

    # Solutions ranked by how *sparsely* covered they are: a solution whose
    # nearest reference vector is far away sits in an under-represented region
    # and is a good seed for a new direction.
    nearest_vec_dist = dist.min(axis=1)               # (N,)
    seed_order = np.argsort(-nearest_vec_dist)        # sparsest first

    inactive_idx = np.where(~active)[0]
    max_replace = int(np.floor(max_replace_ratio * K))
    n_replace = int(min(len(inactive_idx), max_replace, len(F)))
    if n_replace <= 0:
        return W.copy()

    # The reference vectors live in objective space (sine_distance shifts by
    # z_min and scales by the ideal/nadir box before measuring the angle).  A
    # new vector that must lie on the front is therefore simply an actual
    # non-dominated objective point: its angle to that solution is zero, so the
    # vector becomes active and sits on the true (possibly degenerate) manifold.
    W_new = W.copy()
    for slot, sol in zip(inactive_idx[:n_replace], seed_order[:n_replace]):
        W_new[slot] = F[sol]
    return W_new


def repair_reference_vectors(
    W_uniform: np.ndarray,
    H: np.ndarray,
    z_min: np.ndarray,
) -> np.ndarray:
    """Project uniform weight vectors onto the normal hyperplane.

    Each uniform vector ``w`` is a direction in objective space.  Its
    intersection with the hyperplane ``H^T (f - z_min) = 1`` is obtained by
    scaling: find ``t`` such that ``H^T (t * w - z_min) = 1``.  Then
        t = (1 + H^T z_min) / (H^T w).
    The repaired vector is ``t * w``.  Finally, vectors are normalized so
    that the largest component is 1, matching the conventional weight
    scale.

    Parameters
    ----------
    W_uniform : np.ndarray, shape (K, M)
        Uniformly distributed weight vectors (all non-negative, sum ~ 1).
    H : np.ndarray, shape (M,)
        Normal hyperplane vector.
    z_min : np.ndarray, shape (M,)
        Ideal point.

    Returns
    -------
    W_repaired : np.ndarray, shape (K, M)
        Repaired reference vectors lying on the estimated Pareto front.
    """
    W_uniform = np.asarray(W_uniform, dtype=float)
    H = np.asarray(H, dtype=float).reshape(-1)
    z_min = np.asarray(z_min, dtype=float).reshape(-1)

    # Guard against directions parallel to the hyperplane.
    denom = W_uniform @ H
    sign = np.where(denom < 0.0, -1.0, 1.0)
    denom = np.where(np.abs(denom) < 1e-12, sign * 1e-12, denom)

    t = (1.0 + H @ z_min) / denom
    W_repaired = t[:, None] * W_uniform

    # Ensure non-negativity (projection can flip signs for degenerate fronts).
    W_repaired = np.maximum(W_repaired, 1e-12)

    # Normalize to the conventional weight scale (max component = 1).
    max_vals = W_repaired.max(axis=1, keepdims=True)
    max_vals = np.where(max_vals < 1e-12, 1.0, max_vals)
    W_repaired = W_repaired / max_vals

    return W_repaired


def vertical_projection(
    F: np.ndarray,
    H: np.ndarray,
    z_min: np.ndarray,
) -> np.ndarray:
    """Project objective vectors onto the normal hyperplane.

    The projection of a point ``p`` onto the hyperplane
    ``H^T (f - z_min) = 1`` along the normal direction ``H`` is
        f_proj = p - ((H^T(p - z_min) - 1) / (H^T H)) * H.
    This mapping is used by the DiversityArchive to perform niching on the
    repaired front geometry rather than on the raw objective space.

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Objective vectors.
    H : np.ndarray, shape (M,)
        Normal hyperplane vector.
    z_min : np.ndarray, shape (M,)
        Ideal point.

    Returns
    -------
    F_proj : np.ndarray, shape (N, M)
        Vertically projected objective vectors.
    """
    F = np.asarray(F, dtype=float)
    H = np.asarray(H, dtype=float).reshape(1, -1)
    z_min = np.asarray(z_min, dtype=float).reshape(1, -1)

    shifted = F - z_min
    t = (shifted @ H.T - 1.0) / (H @ H.T)
    F_proj = F - t * H

    return F_proj


# ---------------------------------------------------------------------------
# Distance utilities for niching
# ---------------------------------------------------------------------------

def sine_distance(
    F: np.ndarray,
    W: np.ndarray,
    z_min: np.ndarray,
    z_nad: np.ndarray,
) -> np.ndarray:
    """Sine-based angular distance between objective vectors and reference vectors.

    After normalization with respect to the ideal/nadir box, the cosine
    similarity between each ``f`` and each ``w`` is computed.  The sine
    distance is ``sqrt(1 - cos^2)``; it measures how far a solution is from
    a reference direction in angular terms, which is the standard angular
    niching metric for reference-vector association.

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Objective vectors.
    W : np.ndarray, shape (K, M)
        Reference vectors (repaired or uniform).
    z_min : np.ndarray, shape (M,)
        Ideal point.
    z_nad : np.ndarray, shape (M,)
        Nadir point.

    Returns
    -------
    dist : np.ndarray, shape (N, K)
        Sine-based angular distances.
    """
    F = np.asarray(F, dtype=float)
    W = np.asarray(W, dtype=float)
    z_min = np.asarray(z_min, dtype=float).reshape(1, -1)
    z_nad = np.asarray(z_nad, dtype=float).reshape(1, -1)

    denom = np.maximum(z_nad - z_min, 1e-12)
    Fn = (F - z_min) / denom
    Wn = (W - z_min) / denom

    # Avoid division by zero.
    fn_norm = np.linalg.norm(Fn, axis=1, keepdims=True)
    wn_norm = np.linalg.norm(Wn, axis=1, keepdims=True)
    fn_norm = np.where(fn_norm < 1e-12, 1.0, fn_norm)
    wn_norm = np.where(wn_norm < 1e-12, 1.0, wn_norm)

    cosine = (Fn @ Wn.T) / (fn_norm * wn_norm.T)
    cosine = np.clip(cosine, -1.0, 1.0)
    sine = np.sqrt(np.maximum(1.0 - cosine ** 2, 0.0))

    return sine


def perpendicular_distance(
    F: np.ndarray,
    H: np.ndarray,
    z_min: np.ndarray,
) -> np.ndarray:
    """Perpendicular distance from objective vectors to the normal hyperplane.

    This is the convergence indicator used by the ConvergenceArchive: the
    smaller the value, the closer the solution is to the estimated Pareto
    front.

    Parameters
    ----------
    F : np.ndarray, shape (N, M)
        Objective vectors.
    H : np.ndarray, shape (M,)
        Normal hyperplane vector.
    z_min : np.ndarray, shape (M,)
        Ideal point.

    Returns
    -------
    dist : np.ndarray, shape (N,)
        Perpendicular distances.
    """
    F = np.asarray(F, dtype=float)
    H = np.asarray(H, dtype=float).reshape(1, -1)
    z_min = np.asarray(z_min, dtype=float).reshape(1, -1)

    shifted = F - z_min
    num = np.abs(shifted @ H.T - 1.0)
    den = np.linalg.norm(H)
    den = max(den, 1e-12)

    return (num / den).reshape(-1)
