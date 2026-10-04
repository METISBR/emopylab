"""NSGA-III reference-point selection (hyperplane normalisation + niche preservation) for constraint-aware populations."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import cons, nd_sort, objs

__all__ = ["last_selection", "select"]


def last_selection(F1, F2, K, Z, Zmin, rng):
    """Boolean mask over ``F2`` of the ``K`` solutions of the last front that are kept by niche counting."""
    F = np.vstack([F1, F2]) - Zmin
    N, M = F.shape
    N1, N2, NZ = len(F1), len(F2), len(Z)
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    extreme = np.array([np.argmin(np.max(F / w[i], axis=1)) for i in range(M)])
    try:
        a = 1.0 / np.linalg.solve(F[extreme], np.ones(M))
    except np.linalg.LinAlgError:
        a = np.full(M, np.nan)
    if np.any(np.isnan(a)):
        a = F.max(axis=0)
    F = F / a
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = 1 - (F @ Z.T) / (np.linalg.norm(F, axis=1)[:, None] * np.linalg.norm(Z, axis=1)[None, :])
        dist = np.linalg.norm(F, axis=1)[:, None] * np.sqrt(np.maximum(1 - (1 - cosine) ** 2, 0))
    dist = np.where(np.isnan(dist), np.inf, dist)
    pi = np.argmin(dist, axis=1)
    d = dist[np.arange(N), pi]
    rho = np.bincount(pi[:N1], minlength=NZ).astype(float)
    choose = np.zeros(N2, bool)
    zchoose = np.ones(NZ, bool)
    while choose.sum() < K:
        tmp = np.where(zchoose)[0]
        jmin = np.where(rho[tmp] == rho[tmp].min())[0]
        j = tmp[jmin[int(rng.integers(0, len(jmin)))]]
        I = np.where(~choose & (pi[N1:] == j))[0]
        if len(I):
            s = int(np.argmin(d[N1 + I])) if rho[j] == 0 else int(rng.integers(0, len(I)))
            choose[I[s]] = True
            rho[j] += 1
        else:
            zchoose[j] = False
    return choose


def select(pop, N, Z, Zmin, rng):
    """Environmental selection of ``N`` solutions (constrained non-dominated sorting, then niching on the last front)."""
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, N)
    nxt = front < maxf
    last = np.where(front == maxf)[0]
    Zmin = np.ones(Z.shape[1]) if Zmin is None else Zmin
    choose = last_selection(F[nxt], F[last], N - int(nxt.sum()), Z, Zmin, rng)
    nxt[last[choose]] = True
    return pop[nxt]
