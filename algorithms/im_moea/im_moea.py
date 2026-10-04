# emopylab 2026
"""IM-MOEA (inverse modeling based multiobjective evolutionary algorithm).

Reference:
R. Cheng, Y. Jin, K. Narukawa, and B. Sendhoff. A multiobjective evolutionary algorithm using
Gaussian process-based inverse modeling. IEEE Transactions on Evolutionary Computation, 2015, 19(6):
838-856.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, crowding, nd_sort, objs, uniform_point
from algorithms.community_utils.inverse_model import inverse_model_offspring
from core.population import Population

ALGORITHM_FLAGS = {'IMMOEA': {'integer', 'large', 'multi', 'real'}}


def _delete_mask(pop, n_sub):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, n_sub)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-crowding(F[last]), kind="stable")
    nxt[last[rank[: n_sub - int(nxt.sum())]]] = True
    return ~nxt


class IMMOEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, K: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.K = int(K)

    def initial_size(self):
        W, self.K = uniform_point(self.K, self.M)
        self.W = W[np.lexsort(tuple(W[:, j] for j in range(W.shape[1])))]   # sortrows on the reversed columns
        self.pop_size = int(np.ceil(self.pop_size / self.K) * self.K)
        return self.pop_size

    def _partition(self, pop):
        return np.argmin(cosine_distance(objs(pop), self.W), axis=1)

    def start(self):
        self.part = self._partition(self.pop)

    def step(self):
        N0, per = len(self.pop), self.N / self.K
        base = self.pop
        for k in np.unique(self.part):
            child = self.evaluate(inverse_model_offspring(self, base[self.part[:N0] == k]))
            self.pop = Population.merge(self.pop, child)
        self.part = self._partition(self.pop)
        for k in np.unique(self.part):
            cur = np.where(self.part == k)[0]
            if len(cur) > per:
                dele = _delete_mask(self.pop[cur], int(per))
                keep = np.ones(len(self.pop), bool)
                keep[cur[dele]] = False
                self.pop = self.pop[keep]
                self.part = self.part[keep]
