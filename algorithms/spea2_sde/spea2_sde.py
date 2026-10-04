# emopylab 2026
"""SPEA2+SDE (sPEA2 with shift-based density estimation).

Reference:
M. Li, S. Yang, and X. Liu. Shift-based density estimation for Pareto-based algorithms in many-
objective optimization. IEEE Transactions on Evolutionary Computation, 2014, 18(3): 348-365.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, dominance_matrix, ga, objs, sde_distance, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'SPEA2SDE': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def _fitness(F):
    N = len(F)
    dom = dominance_matrix(F)
    S = dom.sum(axis=1)
    R = (dom * S[:, None]).sum(axis=0)                 # sum of strengths of the dominators
    dist = np.sort(sde_distance(F), axis=1)
    return R + 1.0 / (dist[:, int(np.floor(np.sqrt(N))) - 1] + 2.0)


def _environmental_selection(pop, N):
    F = objs(pop)
    fit = _fitness(F)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncate_lexi(sde_distance(F[nxt]), int(nxt.sum()) - N)]] = False
    return pop[nxt], fit[nxt]


class SPEA2SDE(LoopAlgorithm):
    def start(self):
        self.fitness = _fitness(objs(self.pop))

    def step(self):
        pool = tournament(2, self.N, self.fitness, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.fitness = _environmental_selection(Population.merge(self.pop, off), self.N)
