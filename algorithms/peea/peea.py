# emopylab 2026
"""PeEA (pareto front shape estimation based evolutionary algorithm).

Reference:
L. Li, G. G. Yen, A. Sahoo, L. Chang, and T. Gu. On the estimation of pareto front and dimensional
similarity in many-objective evolutionary algorithm. Information Sciences, 2021, 563: 375-400.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, angle_matrix, decs, ga, objs
from core.population import Population

ALGORITHM_FLAGS = {'PeEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _cal_fitness(F):
    N, M = F.shape
    zmin = F.min(axis=0)
    W = np.full((M, M), 1e-6)
    np.fill_diagonal(W, 1.0)
    asf = np.stack([np.max((F - zmin) / W[i], axis=1) for i in range(M)], axis=1)
    extreme = np.argmin(asf, axis=0)
    try:
        hyper = np.linalg.solve(F[extreme], np.ones(M))
        with np.errstate(all="ignore"):
            a = 1.0 / hyper
    except np.linalg.LinAlgError:
        a = np.full(M, np.nan)
    if np.any(np.isnan(a)):
        a = F.max(axis=0)
    with np.errstate(all="ignore"):
        Fn = (F - zmin) / (a - zmin)
    Fn = np.nan_to_num(Fn)
    w_nad = a
    zmin2 = Fn.min(axis=0)
    with np.errstate(all="ignore"):
        col = np.max((Fn - zmin2) / w_nad, axis=1)
    key_point = np.repeat(int(np.argmin(col)), M)                      # every ASF column is identical
    flat = Fn.flatten(order="F")                                       # linear (column-major) indexing
    q = np.sqrt(np.sum(flat[key_point] ** 2)) * np.sqrt(M)
    fx = np.sum(Fn - zmin2, axis=1) if q <= 1 else np.max(Fn - zmin2, axis=1)
    lo, hi = np.minimum(Fn[:, None, :], Fn[None, :, :]), np.maximum(Fn[:, None, :], Fn[None, :, :])
    dist = np.sum((hi - lo) / (1 + lo), axis=2)
    np.fill_diagonal(dist, np.inf)
    dist = np.sort(dist, axis=1)
    k = int(np.floor(np.sqrt(N)))
    return fx + 1.0 / (dist[:, k - 1] + 2.0), extreme


def _mating_selection(F, rng):
    N = len(F)
    fit, _ = _cal_fitness(F)
    p1, p2 = rng.integers(0, N, size=N), rng.integers(0, N, size=N)
    dom = np.any(F[p1] < F[p2], axis=1).astype(int) - np.any(F[p1] > F[p2], axis=1).astype(int)
    return np.concatenate([p1[dom == 1], p2[dom == -1], p1[(dom == 0) & (fit[p1] <= fit[p2])], p2[(dom == 0) & (fit[p1] > fit[p2])]])


def _environmental_selection(pop, T):
    F = objs(pop)
    fit, extreme = _cal_fitness(F)
    extreme = np.unique(extreme)
    angle = angle_matrix(F)
    np.fill_diagonal(angle, np.inf)
    remain = np.delete(np.arange(len(pop)), extreme)
    while len(remain) > T - len(extreme):
        sub = angle[np.ix_(remain, remain)]
        rank1 = np.argsort(sub, axis=1, kind="stable")
        sort_a = np.take_along_axis(sub, rank1, axis=1)
        A = int(np.lexsort(sort_a.T[::-1])[0])
        B = int(rank1[A, 0])
        remain = np.delete(remain, A if fit[remain[A]] > fit[remain[B]] else B)
    return pop[np.concatenate([remain, extreme])]


class PeEA(LoopAlgorithm):
    def step(self):
        pool = _mating_selection(objs(self.pop), self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N)
