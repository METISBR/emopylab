# emopylab 2026
"""MaOEA-CSS (many-objective evolutionary algorithms based on coordinated selection).

Reference:
Z. He and G. G. Yen. Many-objective evolutionary algorithms based on coordinated selection strategy.
IEEE Transactions on Evolutionary Computation, 2017, 21(2): 220-233.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, angle_matrix, decs, ga, objs
from core.population import Population

ALGORITHM_FLAGS = {'MaOEACSS': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _mating_selection(F, zmin, rng):
    N, M = F.shape
    with np.errstate(all="ignore"):
        W = np.maximum(1e-6, F / F.sum(axis=1, keepdims=True))
    G = F - zmin
    with np.errstate(all="ignore"):
        asf = np.max(G / W, axis=1)
    rank = np.argsort(asf, kind="stable")
    asf_rank = np.empty(N, dtype=int)
    asf_rank[rank] = np.arange(1, N + 1)
    angle = angle_matrix(G)
    np.fill_diagonal(angle, np.inf)
    amin = angle.min(axis=1)
    p = np.array([rng.permutation(N)[:2] for _ in range(N)])
    first = (asf[p[:, 0]] < asf[p[:, 1]]) & (amin[p[:, 0]] > amin[p[:, 1]])
    p = np.where(first, p[:, 0], p[:, 1])
    keep = rng.random(N) < 1.0002 - asf_rank[p] / N
    return np.where(keep, p, rng.integers(0, N, size=N))


def _environmental_selection(pop, zmin, t, K):
    G = objs(pop) - zmin
    con = np.sqrt(np.sum(G ** 2, axis=1))
    angle = angle_matrix(G)
    np.fill_diagonal(angle, np.inf)
    remain = np.arange(len(pop))
    while len(remain) > K:
        sub = angle[np.ix_(remain, remain)]
        rank1 = np.argsort(sub, axis=1, kind="stable")
        sort_a = np.take_along_axis(sub, rank1, axis=1)
        A = np.lexsort(sort_a.T[::-1])[0]
        B = rank1[A, 0]
        if con[remain[A]] - con[remain[B]] > t:
            remain = np.delete(remain, A)
        elif con[remain[B]] - con[remain[A]] > t:
            remain = np.delete(remain, B)
        else:
            remain = np.delete(remain, A)
    return pop[remain]


class MaOEACSS(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, t: float = 0.0, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.t = float(t)

    def start(self):
        self.zmin = objs(self.pop).min(axis=0)

    def step(self):
        pool = _mating_selection(objs(self.pop), self.zmin, self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.zmin = np.minimum(self.zmin, objs(off).min(axis=0))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.zmin, self.t, self.N)
