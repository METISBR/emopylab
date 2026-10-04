# emopylab 2026
"""WV-MOEA-P (weight vector based multi-objective optimization algorithm with preference).

Reference:
X. Zhang, X. Jiang, and L. Zhang. A weight vector based multi-objective optimization algorithm with
preference. Acta Electronica Sinica (Chinese), 2016, 44(11): 2639-2645.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, neighbors_of, objs, pdist2, stm_select, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'WVMOEAP': {'integer', 'multi', 'real'}}


class WVMOEAP(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, Points=None, b: float = 0.05, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Points = None if Points is None else np.atleast_2d(np.asarray(Points, dtype=float))
        self.b = float(b)

    def initial_size(self):
        pts = np.ones((1, self.M)) if self.Points is None else self.Points
        self.pts = pts
        W, self.pop_size = uniform_point(self.pop_size, self.M)
        N = self.pop_size
        T = int(np.ceil(N / 10))
        dis = pdist2(W, W)
        self.B = np.zeros((N, T), dtype=int)
        self.group = np.ceil((np.arange(1, N + 1)) / N * len(pts)).astype(int)
        W = W.copy()
        for gi in np.unique(self.group):
            cur = np.where(self.group == gi)[0]
            W[cur] = 2 * self.b * W[cur] + pts[gi - 1] + self.b
            rank = np.argsort(dis[np.ix_(cur, cur)], axis=1, kind="stable")
            self.B[cur] = cur[rank[:, :T]]
        self.W = W
        return N

    def start(self):
        self.Z = objs(self.pop).min(axis=0)

    def step(self):
        rng, T = self.rng, self.B.shape[1]
        for gi in np.unique(self.group):
            cur = np.where(self.group == gi)[0]
            P = np.zeros((len(cur), 2), dtype=int)
            for j in range(len(cur)):
                P[j] = self.B[j, rng.permutation(T)[:2]] if rng.random() < 0.9 else cur[rng.permutation(len(cur))[:2]]
            off = self.evaluate(de(self.problem, decs(self.pop[cur]), decs(self.pop[P[:, 0]]), decs(self.pop[P[:, 1]]), rng=rng))
            self.Z = np.minimum(self.Z, objs(off).min(axis=0))
            merged = Population.merge(self.pop[cur], off)
            sel = stm_select(objs(merged), self.W[cur], self.Z, self.Z + 1.0, rng)
            self.pop[cur] = merged[sel]
