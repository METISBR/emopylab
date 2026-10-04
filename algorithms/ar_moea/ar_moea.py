# emopylab 2026
"""AR-MOEA (Adaptive reference-point based MOEA).

Reference:
Y. Tian, R. Cheng, X. Zhang, and Y. Jin. An indicator-based multiobjective evolutionary algorithm with reference point adaptation for better versatility. IEEE Transactions on Evolutionary Computation, 2018, 22(4): 609-622.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.armoea import contribution_fitness, last_selection, update_ref_point
from algorithms.community_utils.base import LoopAlgorithm, cv, decs, ga, nd_sort, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'ARMOEA': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class ARMOEA(LoopAlgorithm):
    """Indicator-based MOEA with adaptive reference points: an archive of non-dominated feasible solutions reshapes the
    reference points, parents are chosen by constraint violation and indicator contribution, and the last front is
    truncated by removing the solution whose loss hurts the indicator least."""

    ALGO_FLAGS = {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}
    OBJECTIVE_SCOPE = "many"

    def _feasible_objs(self, pop):
        return objs(pop)[cv(pop) <= 0] if len(pop) else np.zeros((0, self.M))

    def start(self):
        self.W = uniform_point(self.N, self.M)[0]
        self.archive, self.ref, self.range, _ = update_ref_point(self._feasible_objs(self.pop), self.W, None)

    def _mating(self):
        pop = self.pop
        CV = cv(pop)
        feas = CV == 0
        if feas.sum() > 1:
            fit = contribution_fitness(objs(pop)[feas], self.ref, self.range)
        else:
            fit = np.zeros(int(feas.sum()))
        Fitness = np.full(len(pop), -np.inf)
        Fitness[feas] = fit
        return tournament(2, len(pop), CV, -Fitness, rng=self.rng)

    def _environmental_selection(self, pop):
        N = self.N
        CV = cv(pop)
        if np.sum(CV == 0) > N:
            pop = pop[CV == 0]
            F = objs(pop)
            front, max_f = nd_sort(F, None, N)
            nxt = front < max_f
            last = np.where(front == max_f)[0]
            choose = last_selection(F[last], self.ref, self.range, N - int(nxt.sum()))
            nxt[last[choose]] = True
            pop = pop[nxt]
            self.range = np.array(self.range, dtype=float)
            self.range[1] = objs(pop).max(axis=0)
            self.range[1, self.range[1] - self.range[0] < 1e-6] = 1
            return pop
        return pop[np.argsort(CV, kind="stable")[:N]]

    def step(self):
        off = self.evaluate(ga(self.problem, decs(self.pop[self._mating()]), rng=self.rng))
        A = self._feasible_objs(off)
        A = np.vstack([self.archive, A]) if np.size(self.archive) else A
        self.archive, self.ref, self.range, _ = update_ref_point(A, self.W, self.range)
        self.pop = self._environmental_selection(Population.merge(self.pop, off))
