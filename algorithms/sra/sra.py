# emopylab 2026
"""SRA (stochastic ranking algorithm).

Reference:
B. Li, K. Tang, J. Li, and X. Yao. Stochastic ranking algorithm for many-objective optimization
based on multiple indicators. IEEE Transactions on Evolutionary Computation, 2016, 20(6): 924-938.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga_half, objs
from core.population import Population

ALGORITHM_FLAGS = {'SRA': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def _environmental_selection(pop, K, pc, rng):
    N = len(pop)
    F = objs(pop)
    I = np.max(F[:, None, :] - F[None, :, :], axis=2)                 # I[i, j] = max_m (f_i - f_j)
    I1 = np.sum(-np.exp(-I / 0.05), axis=0) + 1.0
    Distance = np.full((N, N), np.inf)
    for i in range(N):
        S = np.maximum(F, F[i])                                       # shifted objectives, rows j
        if i:
            Distance[i, :i] = np.linalg.norm(F[i] - S[:i], axis=1)
    I2 = Distance.min(axis=1)
    rank = list(range(N))
    for _ in range(int(np.ceil(N / 2))):
        swapped = False
        for j in range(N - 1):
            ind = I1 if rng.random() < pc else I2
            if ind[rank[j]] < ind[rank[j + 1]]:
                rank[j], rank[j + 1] = rank[j + 1], rank[j]
                swapped = True
        if not swapped:
            break
    return pop[np.asarray(rank[:K])]


class SRA(LoopAlgorithm):
    def step(self):
        pool = self.rng.integers(0, len(self.pop), size=2 * self.N)
        off = self.evaluate(ga_half(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N, self.rng.uniform(0.4, 0.6), self.rng)
