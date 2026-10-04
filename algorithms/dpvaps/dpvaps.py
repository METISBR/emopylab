# emopylab 2026
"""DPVAPS (dual-population with variable auxiliary population size).

Reference:
J. Liang, Z. Chen, Y. Wang, X. Ban, K. Qiao, and K. Yu. A dual-population constrained multi-
objective evolutionary algorithm with variable auxiliary population size. Complex & Intelligent
Systems, 2023, 9: 5907-5922.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import cal_fitness, environmental_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, ga_half, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'DPVAPS': {'constrained', 'integer', 'large', 'multi', 'real'}}


class DPVAPS(LoopAlgorithm):
    """Dual populations with an archive of the feasible solutions the helper population discovered; the helper
    population shrinks linearly as the search advances."""

    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type

    def start(self):
        self.pop1 = self.pop
        self.pop2 = self.evaluate(self.random_decs(self.N))
        self.fit1 = cal_fitness(objs(self.pop1), cons(self.pop1))
        self.fit2 = cal_fitness(objs(self.pop2))
        self.A = None

    def step(self):
        N, rng, P1, P2 = self.N, self.rng, self.pop1, self.pop2
        n2 = len(P2)
        if self.type == 1:
            m1, m2 = tournament(2, N, self.fit1, rng=rng), tournament(2, n2, self.fit2, rng=rng)
            off1 = self.evaluate(ga_half(self.problem, decs(P1[m1]), rng=rng))
            off2 = self.evaluate(ga_half(self.problem, decs(P2[m2]), rng=rng))
        else:
            m1, m2 = tournament(2, 2 * N, self.fit1, rng=rng), tournament(2, 2 * n2, self.fit2, rng=rng)
            off1 = self.evaluate(de(self.problem, decs(P1), decs(P1[m1[:N]]), decs(P1[m1[N:]]), rng=rng))
            off2 = self.evaluate(de(self.problem, decs(P2), decs(P2[m2[:n2]]), decs(P2[m2[n2:]]), rng=rng))
        C2 = cons(P2)
        feas = np.sum(np.maximum(0, C2), axis=1) <= 0 if C2.size else np.ones(n2, bool)
        self.A = Population.merge(self.A, P2[feas])
        if len(self.A) > N:
            self.A, _ = environmental_selection(self.A, N, True)
        self.pop1, self.fit1 = environmental_selection(Population.merge(P1, off1, self.A, off2), N, True)
        delta = -0.9 * (self.FE / self.max_FE) + 1
        self.pop2, self.fit2 = environmental_selection(Population.merge(P2, off1, off2), int(np.floor(N * delta + 0.5)), False)
        self.pop = self.pop1
