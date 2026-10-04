"""SPEA2-style strength/density fitness with (epsilon-)constraint handling, shared by several constrained algorithms."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import cons, objs, truncate_lexi

__all__ = ["cal_fitness", "truncation", "select", "overall_cv"]


def overall_cv(C) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    return np.sum(np.maximum(0.0, C), axis=1) if C.size else np.zeros(len(C))


def cal_fitness(F, C=None, epsilon=None) -> np.ndarray:
    """Raw strength fitness plus k-th nearest neighbour density (< 1 means non-dominated).  Constraint violations below
    ``epsilon`` (strictly) count as feasible; a violation ranks a solution behind every less-violating one."""
    F = np.asarray(F, dtype=float)
    N = len(F)
    CV = np.zeros(N) if C is None else overall_cv(C)
    if epsilon is not None:
        CV = np.where(CV < epsilon, 0.0, CV)
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    Dom = (CV[:, None] < CV[None, :]) | ((CV[:, None] == CV[None, :]) & (k == 1))
    S = Dom.sum(axis=1)
    R = S @ Dom
    dist = np.sqrt(np.maximum(np.sum(F * F, 1)[:, None] + np.sum(F * F, 1)[None, :] - 2.0 * F @ F.T, 0.0))
    np.fill_diagonal(dist, np.inf)
    dist = np.sort(dist, axis=1)
    return R + 1.0 / (dist[:, max(int(np.floor(np.sqrt(N))) - 1, 0)] + 2)


def truncation(F, K) -> np.ndarray:
    """Boolean mask of the ``K`` solutions removed by iterated nearest-neighbour truncation."""
    F = np.asarray(F, dtype=float)
    dist = np.sqrt(np.maximum(np.sum(F * F, 1)[:, None] + np.sum(F * F, 1)[None, :] - 2.0 * F @ F.T, 0.0))
    np.fill_diagonal(dist, np.inf)
    return truncate_lexi(dist, K)


def select(pop, N, use_cons=True, epsilon=None):
    """Environmental selection on the fitness above (non-dominated first, then best ranks, or truncation).

    Returns ``(pop[Next], fitness[Next])`` keeping the original order."""
    F = objs(pop)
    fit = cal_fitness(F, cons(pop) if use_cons else None, epsilon)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt], fit[nxt]
