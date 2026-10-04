# emopylab 2026
"""TiGE-2 (tri-Goal Evolution Framework for CMaOPs).

Reference:
Y. Zhou, Z. Min, J. Wang, Z. Zhang, and J.Zhang. Tri-goal evolution framework for constrained many-
objective optimization. IEEE Transactions on Systems Man and Cybernetics Systems, 2020, 50(8):
3086-3099.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament
from algorithms.bige.bige import _estimation
from core.population import Population

ALGORITHM_FLAGS = {'TiGE2': {'binary', 'constrained', 'integer', 'label', 'many', 'permutation', 'real'}}


def _fcv(pop):
    c = cons(pop)
    if c.size == 0:
        return np.zeros(len(pop))
    c = np.where(c <= 0, 0.0, c)
    with np.errstate(all="ignore"):
        c = c / c.max(axis=0)
    c = np.where(np.isnan(c), 0.0, c)
    return np.maximum(0.0, c).sum(axis=1) / c.shape[1]


def _tri_objective(pop, N, M, eps):
    """(front rank of the bi-goal estimation + eps * fcv, fcv) as the two objectives ranked by NSGA-II."""
    est = _estimation(objs(pop), 1.0 / N ** (1.0 / M))
    fcv = _fcv(pop)
    fm, _ = nd_sort(est, None, N)
    return np.column_stack([fm + eps * fcv, fcv]), fcv


def _environmental_selection(pop, pop_obj, off_obj, N):
    front_no, _ = nd_sort(np.vstack([pop_obj, off_obj]), None, N)
    fcv = _fcv(pop)
    fitness = front_no + fcv / (fcv + 1)
    idx = np.argsort(fitness, kind="stable")[:N]
    return pop[idx], fitness[idx]


class TiGE2(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, Epsilon0: float = 0.05, row: float = 1.01, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.epsilon0 = float(Epsilon0)
        self.row = float(row)

    def start(self):
        self.eps = self.epsilon0
        self.pop_obj, fcv = _tri_objective(self.pop, self.N, self.M, self.eps)
        frank, _ = nd_sort(self.pop_obj, None, self.N)
        self.fitness = frank + fcv / (fcv + 1)

    def step(self):
        pool = tournament(2, self.N, self.fitness, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        off_obj, _ = _tri_objective(off, self.N, self.M, self.eps)
        self.pop, self.fitness = _environmental_selection(Population.merge(self.pop, off), self.pop_obj, off_obj, self.N)
        self.pop_obj, _ = _tri_objective(self.pop, self.N, self.M, self.eps)
        self.eps *= self.row
