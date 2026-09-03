from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def _safe_span(zmin: np.ndarray, zmax: np.ndarray) -> np.ndarray:
    span = np.asarray(zmax, dtype=float).reshape(-1) - np.asarray(zmin, dtype=float).reshape(-1)
    return np.where(np.abs(span) <= 1e-12, 1.0, span)


def _row_norms(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=float), axis=1)


def deduplicate_objective_indices(F: np.ndarray, cv: np.ndarray, *, decimals: int = 10) -> np.ndarray:
    """Keep one representative per rounded objective vector, preferring lower CV."""
    F = np.asarray(F, dtype=float)
    cv = np.asarray(cv, dtype=float).reshape(-1)
    if F.size == 0:
        return np.empty(0, dtype=int)

    rounded = np.round(F, decimals=int(decimals))
    best: Dict[tuple[float, ...], int] = {}

    for idx in range(len(rounded)):
        key = tuple(float(v) for v in rounded[idx])
        prev = best.get(key)
        if prev is None or cv[idx] < cv[prev]:
            best[key] = idx

    kept = np.asarray(sorted(best.values()), dtype=int)
    return kept


def _last_selection(
    pop_obj_1: np.ndarray,
    pop_obj_2: np.ndarray,
    k: int,
    ref_dirs: np.ndarray,
    zmin: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    NSGA-III final niching phase adapted from algorithms/nsga3_local/nsga3_local.py.
    Returns a boolean mask over pop_obj_2.
    """
    n2 = int(pop_obj_2.shape[0])
    choose = np.zeros(n2, dtype=bool)
    if n2 == 0 or int(k) <= 0:
        return choose

    k = int(max(0, min(int(k), n2)))
    pop_obj_1 = np.asarray(pop_obj_1, dtype=float)
    pop_obj_2 = np.asarray(pop_obj_2, dtype=float)
    ref_dirs = np.asarray(ref_dirs, dtype=float)
    zmin = np.asarray(zmin, dtype=float).reshape(-1)

    if ref_dirs.ndim != 2 or ref_dirs.shape[0] == 0:
        choose[:k] = True
        return choose

    pop_obj = np.vstack([pop_obj_1, pop_obj_2]) - zmin[None, :]
    n, m = pop_obj.shape
    n1 = int(pop_obj_1.shape[0])
    n_ref = int(ref_dirs.shape[0])

    # ASF-based extreme point detection.
    w = np.eye(m, dtype=float) + 1e-6
    extreme = np.zeros(m, dtype=int)
    for i in range(m):
        with np.errstate(divide="ignore", invalid="ignore"):
            asf = np.max(pop_obj / w[i][None, :], axis=1)
        asf = np.where(np.isfinite(asf), asf, np.inf)
        extreme[i] = int(np.argmin(asf))

    intercept = None
    try:
        intercept = np.linalg.lstsq(pop_obj[extreme, :], np.ones(m, dtype=float), rcond=None)[0]
    except Exception:
        intercept = None

    if intercept is None:
        a = np.max(pop_obj, axis=0)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = 1.0 / intercept
        if np.any(~np.isfinite(a)) or np.any(a <= 1e-12):
            a = np.max(pop_obj, axis=0)

    a = np.where(np.isfinite(a) & (a > 1e-12), a, 1.0)
    pop_obj = pop_obj / a[None, :]

    pop_norm = _row_norms(pop_obj)
    ref_norm = _row_norms(ref_dirs)
    denom = pop_norm[:, None] * ref_norm[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.divide(
            pop_obj @ ref_dirs.T,
            denom,
            out=np.zeros((n, n_ref), dtype=float),
            where=denom > 0.0,
        )
    cosine = np.clip(cosine, -1.0, 1.0)
    distance = pop_norm[:, None] * np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))

    pi = np.argmin(distance, axis=1).astype(int, copy=False)
    d = distance[np.arange(n), pi]
    rho = np.bincount(pi[:n1], minlength=n_ref).astype(int, copy=False)

    zchoose = np.ones(n_ref, dtype=bool)
    while int(np.sum(choose)) < k:
        active_refs = np.where(zchoose)[0]
        if active_refs.size == 0:
            break

        active_rho = rho[active_refs]
        min_rho = int(np.min(active_rho))
        tied = active_refs[active_rho == min_rho]
        if tied.size == 1:
            j = int(tied[0])
        else:
            j = int(tied[int(rng.integers(0, tied.size))])

        candidate = np.where((~choose) & (pi[n1:] == j))[0]
        if candidate.size > 0:
            if rho[j] == 0:
                pick_local = int(np.argmin(d[n1 + candidate]))
            else:
                pick_local = int(rng.integers(0, candidate.size))
            choose[int(candidate[pick_local])] = True
            rho[j] += 1
        else:
            zchoose[j] = False

    remaining_needed = int(k - np.sum(choose))
    if remaining_needed > 0:
        remaining_idx = np.where(~choose)[0]
        if remaining_idx.size > 0:
            order = np.argsort(d[n1 + remaining_idx], kind="mergesort")
            pick = remaining_idx[order[:remaining_needed]]
            choose[pick] = True

    return choose


def nsga3_niching_indices(
    F: np.ndarray,
    target_size: int,
    ref_dirs: np.ndarray,
    zmin: np.ndarray,
    zmax: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select exactly target_size indices by NSGA-III reference niching.
    """
    F = np.asarray(F, dtype=float)
    n = int(F.shape[0])
    target = int(max(0, target_size))

    if target == 0 or n == 0:
        return np.empty(0, dtype=int)
    if n <= target:
        return np.arange(n, dtype=int)

    span = _safe_span(zmin=np.asarray(zmin, dtype=float), zmax=np.asarray(zmax, dtype=float))
    Fn = (F - np.asarray(zmin, dtype=float)[None, :]) / span[None, :]

    W = np.asarray(ref_dirs, dtype=float)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] != Fn.shape[1]:
        norms = _row_norms(Fn)
        return np.argsort(norms, kind="mergesort")[:target]

    chosen_mask = _last_selection(
        np.empty((0, Fn.shape[1]), dtype=float),
        Fn,
        target,
        W,
        np.zeros(Fn.shape[1], dtype=float),
        rng,
    )
    idx = np.where(chosen_mask)[0]

    if idx.size < target:
        remain = np.setdiff1d(np.arange(n, dtype=int), idx, assume_unique=False)
        if remain.size > 0:
            norms = _row_norms(Fn[remain])
            fill = remain[np.argsort(norms, kind="mergesort")[: target - idx.size]]
            idx = np.concatenate([idx, fill])

    if idx.size > target:
        idx = idx[:target]

    return idx.astype(int, copy=False)


__all__ = [
    "deduplicate_objective_indices",
    "nsga3_niching_indices",
]
