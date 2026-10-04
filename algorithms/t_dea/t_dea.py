# emopylab 2026
"""t-DEA (theta-dominance based evolutionary algorithm).

Reference:
Y. Yuan, H. Xu, B. Wang, and X. Yao. A new dominance relation-based evolutionary algorithm for many-
objective optimization. IEEE Transactions on Evolutionary Computation, 2016, 20(1): 16-37.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, cosine_distance, decs, first_index_reaching, ga, nd_sort,
                                            normalization, objs, uniform_point)
from core.population import Population

ALGORITHM_FLAGS = {'tDEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _t_nd_sort(F, W):
    N, NW = F.shape[0], len(W)
    norm_p = np.linalg.norm(F, axis=1)
    cosine = 1.0 - cosine_distance(F, W)
    d1 = norm_p[:, None] * cosine
    d2 = norm_p[:, None] * np.sqrt(np.maximum(0.0, 1.0 - cosine ** 2))
    cls = np.argmin(d2, axis=1)
    theta = np.full(NW, 5.0)
    theta[np.sum(W > 1e-4, axis=1) == 1] = 1e6
    t_front = np.zeros(N)
    for i in range(NW):
        C = np.where(cls == i)[0]
        rank = np.argsort(d1[C, i] + theta[i] * d2[C, i], kind="stable")
        t_front[C[rank]] = np.arange(1, len(C) + 1)
    return t_front


def _environmental_selection(pop, W, N, z, znad, rng):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    St = np.where(front_no <= max_f)[0]
    Fn, z, znad = normalization(F[St], z, znad)
    t_front = _t_nd_sort(Fn, W)
    max_t = first_index_reaching(t_front, N)
    last = np.where(t_front == max_t)[0]
    last = last[rng.permutation(len(last))]
    t_front = t_front.copy()
    t_front[last[: int(np.sum(t_front <= max_t)) - N]] = np.inf
    return pop[St[t_front <= max_t]], z, znad


class tDEA(LoopAlgorithm):
    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        self.z, self.znad = F.min(axis=0), F.max(axis=0)

    def step(self):
        pool = self.rng.integers(0, len(self.pop), size=self.N)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.z, self.znad = _environmental_selection(
            Population.merge(self.pop, off), self.W, self.N, self.z, self.znad, self.rng)
