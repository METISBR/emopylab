# emopylab 2026
"""LMOCSO (large-scale multi-objective competitive swarm optimization algorithm).

Reference:
Y. Tian, X. Zheng, X. Zhang, and Y. Jin. Efficient large-scale multi- objective optimization based
on a competitive swarm optimizer. IEEE Transactions on Cybernetics, 2020, 50(8): 3696-3708.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, cons, cv, decs, first_front, nd_sort, objs, pdist2, polynomial_mutation,
                                            uniform_point, velocity)
from core.population import Population

ALGORITHM_FLAGS = {'LMOCSO': {'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


def _angle(A, B):
    """Angle between every row of A and every row of B (NaN for a zero vector, like a cosine distance would give)."""
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    with np.errstate(all="ignore"):
        cos = (A @ B.T) / (na[:, None] * nb[None, :])
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _nan_argmin(x, axis):
    return np.argmin(np.where(np.isnan(x), np.inf, x), axis=axis)


def environmental_selection(pop, V, theta):
    pop = pop[nd_sort(objs(pop), None, 1)[0] == 1]
    F = objs(pop)
    N, M = F.shape
    NV = len(V)
    F = F - F.min(axis=0)
    CV = cv(pop)
    cosine = _angle(V, V)
    cosine = np.cos(cosine)
    np.fill_diagonal(cosine, 0.0)
    gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
    Angle = _angle(F, V)
    associate = _nan_argmin(Angle, 1)
    nxt = -np.ones(NV, dtype=int)
    for i in np.unique(associate):
        c1 = np.where((associate == i) & (CV == 0))[0]
        c2 = np.where((associate == i) & (CV != 0))[0]
        if len(c1):
            with np.errstate(all="ignore"):
                apd = (1 + M * theta * Angle[c1, i] / gamma[i]) * np.sqrt(np.sum(F[c1] ** 2, axis=1))
            nxt[i] = c1[_nan_argmin(apd, 0)]
        elif len(c2):
            nxt[i] = c2[int(np.argmin(CV[c2]))]
    return pop[nxt[nxt >= 0]]


def cal_fitness(F):
    N = len(F)
    with np.errstate(all="ignore"):
        F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        # || F_i - max(F_i, F_j) || = || max(F_j - F_i, 0) ||
        dis = np.linalg.norm(np.maximum(F[None, :, :] - F[:, None, :], 0.0), axis=2)
    np.fill_diagonal(dis, np.inf)
    return np.nanmin(np.where(np.isnan(dis), np.nan, dis), axis=1) if N > 1 else np.full(N, np.inf)


class LMOCSO(LoopAlgorithm):
    """Large-scale multi-objective competitive swarm optimizer with angle-penalized selection."""

    def initial_size(self):
        self.V, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def _theta(self):
        return (self.FE / self.max_FE) ** 2

    def start(self):
        self.pop = environmental_selection(self.pop, self.V, self._theta())

    def _operator(self, loser, winner):
        rng, lo, up = self.rng, self.lower, self.upper
        LD, WD = decs(loser), decs(winner)
        N, D = LD.shape
        LV, WV = velocity(loser), velocity(winner)
        r1 = np.repeat(rng.random((N, 1)), D, axis=1)
        r2 = np.repeat(rng.random((N, 1)), D, axis=1)
        vel = r1 * LV + r2 * (WD - LD)
        dec = LD + vel + r1 * (vel - LV)
        dec = np.vstack([dec, WD])
        vel = np.vstack([vel, WV])
        dec = polynomial_mutation(dec, lo, up, rng)
        return self.evaluate(dec, V=vel)

    def step(self):
        rng, pop = self.rng, self.pop
        fit = cal_fitness(objs(pop))
        n = len(pop)
        rank = rng.permutation(n)[: (n // 2) * 2] if n >= 2 else np.array([0, 0])
        loser, winner = rank[: len(rank) // 2].copy(), rank[len(rank) // 2:].copy()
        change = fit[loser] >= fit[winner]
        loser[change], winner[change] = winner[change].copy(), loser[change].copy()
        off = self._operator(pop[loser], pop[winner])
        self.pop = environmental_selection(Population.merge(pop, off), self.V, self._theta())
