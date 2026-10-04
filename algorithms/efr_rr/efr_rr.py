# emopylab 2026
"""EFR-RR (ensemble fitness ranking with a ranking restriction scheme).

Reference:
Y. Yuan, H. Xu, B. Wang, B. Zhang, and X. Yao. Balancing convergence and diversity in decomposition-
based many-objective optimizers. IEEE Transactions on Evolutionary Computation, 2016, 20(2):
180-198.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, cosine_distance, decs, first_index_reaching, ga, normalization,
                                            objs, tournament, uniform_point)
from core.population import Population

ALGORITHM_FLAGS = {'EFRRR': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _maximum_ranking(F, W, K):
    N, NW = len(F), len(W)
    sim = 1.0 - cosine_distance(F, W)
    L = np.argsort(-sim, axis=1, kind="stable")[:, :K]
    g = np.full((N, NW), np.inf)
    with np.errstate(all="ignore"):
        for i in range(N):
            g[i, L[i]] = np.max(F[i] / W[L[i]], axis=1)
    rank = np.argsort(g, axis=0, kind="stable")
    r = np.argsort(rank, axis=0, kind="stable") + 1                    # r[i, j]: position of i for weight j
    rg = np.min(np.where(np.isfinite(g), r, np.inf), axis=1)
    return np.unique(rg, return_inverse=True)[1] + 1.0


def _environmental_selection(pop, W, N, K, z, znad, rng):
    Fn, z, znad = normalization(objs(pop), z, znad)
    rg = _maximum_ranking(Fn, W, K)
    max_f = first_index_reaching(rg, N)
    last = np.where(rg == max_f)[0]
    last = last[rng.permutation(len(last))]
    rg = rg.copy()
    rg[last[: int(np.sum(rg <= max_f)) - N]] = np.inf
    return pop[rg <= max_f], z, znad


class EFRRR(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, K: int = 2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.K = int(K)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        Fn, self.z, self.znad = normalization(F, F.min(axis=0), F.max(axis=0))
        # the reference implementation keeps the ranking computed at initialisation for mating
        self.rank = _maximum_ranking(Fn, self.W, self.K)

    def step(self):
        pool = tournament(2, self.N, self.rank, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.z, self.znad = _environmental_selection(
            Population.merge(self.pop, off), self.W, self.N, self.K, self.z, self.znad, self.rng)
