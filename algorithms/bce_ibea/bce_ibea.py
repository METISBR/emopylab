# emopylab 2026
"""BCE-IBEA (bi-criterion evolution based IBEA).

Reference:
M. Li, S. Yang, and X. Liu. Pareto or non-Pareto: Bi-criterion evolution in multiobjective
optimization. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 645-665.
"""

from __future__ import annotations

import numpy as np

from algorithms.bce_moead.bce_moead import _pc_selection, exploration
from algorithms.community_utils.base import LoopAlgorithm, decs, ga, objs, tournament
from algorithms.ibea.ibea import cal_fitness, truncate_by_fitness
from core.population import Population

ALGORITHM_FLAGS = {'BCEIBEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class BCEIBEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, kappa: float = 0.05, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.kappa = float(kappa)

    def start(self):
        self.NPC = self.pop
        self.PC, self.nND = _pc_selection(self.NPC, self.N, self.rng)
        self.pop = self.PC

    def step(self):
        rng, N, kappa = self.rng, self.N, self.kappa
        new_pc = exploration(self, self.PC, self.NPC, self.nND, N)
        self.NPC = truncate_by_fitness(self.NPC if new_pc is None else Population.merge(self.NPC, new_pc), N, kappa)
        fit, _, _ = cal_fitness(objs(self.NPC), kappa)
        pool = tournament(2, N, -fit, rng=rng)
        new_npc = self.evaluate(ga(self.problem, decs(self.NPC[pool]), rng=rng))
        self.NPC = truncate_by_fitness(Population.merge(self.NPC, new_npc), N, kappa)
        merged = Population.merge(self.PC, new_npc)
        if new_pc is not None:
            merged = Population.merge(merged, new_pc)
        self.PC, self.nND = _pc_selection(merged, N, rng)
        self.pop = self.PC
