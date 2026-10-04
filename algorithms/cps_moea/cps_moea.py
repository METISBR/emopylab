# emopylab 2026
"""CPS-MOEA (classification and Pareto domination based multi-objective evolutionary).

Reference:
J. Zhang, A. Zhou, and G. Zhang. A classification and Pareto domination based multiobjective
evolutionary algorithm. Proceedings of the IEEE Congress on Evolutionary Computation, 2015,
2883-2890.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, de, decs, first_front, nd_sort, objs, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'CPSMOEA': {'integer', 'multi', 'real'}}


def _nds(pop, K):
    """The K best (front rank, then crowding) and K worst individuals of ``pop``."""
    F = objs(pop)
    front_no, _ = nd_sort(F, None, np.inf)
    cd = crowding(F, front_no)
    rank = np.lexsort((-cd, front_no))
    return pop[rank[:K]], pop[rank[len(rank) - K:]]


class CPSMOEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, M: int = 3, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.M_cand = int(M)

    def start(self):
        self.good, self.bad = _nds(self.pop, self.N // 2)

    def _knn_good(self, cand):
        data = np.vstack([decs(self.good), decs(self.bad)])
        label = np.concatenate([np.ones(len(self.good), bool), np.zeros(len(self.bad), bool)])
        rank = np.argsort(pdist2(cand, data), axis=1, kind="stable")[:, :5]
        return label[rank].sum(axis=1) > 2

    def step(self):
        rng, N, Mc = self.rng, len(self.pop), self.M_cand
        parent1 = np.tile(np.arange(N), Mc)
        cand = de(self.problem, decs(self.pop[parent1]), decs(self.pop[rng.integers(0, N, size=N * Mc)]),
                  decs(self.pop[rng.integers(0, N, size=N * Mc)]), rng=rng)
        labels = self._knn_good(cand).reshape(Mc, N).T + rng.random((N, Mc))
        best = np.argmax(labels, axis=1)
        off = self.evaluate(cand[best * N + np.arange(N)])
        self.pop, _ = _nds(Population.merge(self.pop, off), self.N)
        nd = first_front(objs(off))
        # the reference implementation keeps the "good half" of both merged sets
        self.good = _nds(Population.merge(self.good, off[nd]), self.N // 2)[0]
        self.bad = _nds(Population.merge(self.bad, off[~nd]), self.N // 2)[0]
