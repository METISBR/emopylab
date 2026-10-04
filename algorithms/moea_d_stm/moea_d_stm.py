# emopylab 2026
"""MOEA-D-STM (mOEA/D with stable matching).

Reference:
K. Li, Q. Zhang, S. Kwong, M. Li, and R. Wang. Stable matching-based selection in evolutionary
multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2014, 18(6): 909-923.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, neighbors_of, objs, stm_select, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MOEADSTM': {'integer', 'many', 'multi', 'real'}}


class MOEADSTM(LoopAlgorithm):
    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        self.z = F.min(axis=0)
        self.Pi = np.ones(self.N)
        with np.errstate(all="ignore"):
            self.old_obj = np.max(np.abs((F - self.z) / self.W), axis=1)

    def step(self):
        N, T, rng, W = self.N, self.T, self.rng, self.W
        boundary = np.where(np.sum(W < 1e-3, axis=1) == self.M - 1)[0]
        for _ in range(5):
            I = np.concatenate([boundary, tournament(10, N // 5 - len(boundary), -self.Pi, rng=rng)]).astype(int)
            P = np.zeros((len(I), 3), dtype=int)
            for i, s in enumerate(I):
                P[i] = self.B[s, rng.permutation(T)[:3]] if rng.random() < 0.9 else rng.permutation(N)[:3]
            off = self.evaluate(de(self.problem, decs(self.pop[P[:, 0]]), decs(self.pop[P[:, 1]]), decs(self.pop[P[:, 2]]), rng=rng))
            self.z = np.minimum(self.z, objs(off).min(axis=0))
            merged = Population.merge(self.pop, off)
            self.pop = merged[stm_select(objs(merged), W, self.z, objs(self.pop).max(axis=0), rng)]
        if int(np.ceil(self.FE / N)) % 10 == 0:
            with np.errstate(all="ignore"):
                new_obj = np.max(np.abs((objs(self.pop) - self.z) / W), axis=1)
            delta = self.old_obj - new_obj
            temp = delta < 0.001
            self.Pi[~temp] = 1
            self.Pi[temp] = (0.95 + 0.05 * delta[temp] / 0.001) * self.Pi[temp]
            self.old_obj = new_obj
