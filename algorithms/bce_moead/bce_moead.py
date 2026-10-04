# emopylab 2026
"""BCE-MOEA-D (bi-criterion evolution based MOEA/D).

Reference:
M. Li, S. Yang, and X. Liu. Pareto or non-Pareto: Bi-criterion evolution in multiobjective
optimization. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 645-665.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, decs, first_front, ga_half, neighbors_of, objs, pdist2,
                                            uniform_point)
from core.population import Population

ALGORITHM_FLAGS = {'BCEMOEAD': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _pc_selection(PC, N, rng):
    PC = PC[first_front(objs(PC))]
    PC = PC[rng.permutation(len(PC))]
    nND = len(PC)
    if len(PC) > N:
        F = objs(PC)
        with np.errstate(all="ignore"):
            F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        d = pdist2(F, F)
        np.fill_diagonal(d, np.inf)
        sd = np.sort(d, axis=1)
        r = sd[:, min(3, sd.shape[1]) - 1].mean()
        R = np.minimum(d / r, 1.0)
        alive = np.arange(len(PC))
        while len(alive) > N:
            worst = int(np.argmax(1 - np.prod(R, axis=1)))
            alive = np.delete(alive, worst)
            R = np.delete(np.delete(R, worst, axis=0), worst, axis=1)
        PC = PC[alive]
    return PC, nND


def exploration(algo, PC, NPC, nND, N):
    """Offspring for the under-explored Pareto-criterion members (few non-Pareto neighbours within radius r)."""
    PCf, NPCf = objs(PC), objs(NPC)
    fmax, fmin = PCf.max(axis=0), PCf.min(axis=0)
    with np.errstate(all="ignore"):
        PCn, NPCn = (PCf - fmin) / (fmax - fmin), (NPCf - fmin) / (fmax - fmin)
    d = pdist2(PCn, PCn)
    np.fill_diagonal(d, np.inf)
    d = np.sort(d, axis=1)
    r0 = d[:, min(3, d.shape[1]) - 1].mean()
    r = nND / N * r0
    S = np.where((pdist2(PCn, NPCn) <= r).sum(axis=1) <= 1)[0]
    if len(S) == 0:
        return None
    mating = algo.rng.integers(0, len(PC), size=len(S))
    return algo.evaluate(ga_half(algo.problem, np.vstack([decs(PC[S]), decs(PC[mating])]), rng=algo.rng))


class BCEMOEAD(LoopAlgorithm):
    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.nr = int(np.ceil(self.pop_size / 100))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.NPC = self.pop
        self.Z = objs(self.NPC).min(axis=0)
        self.PC, self.nND = _pc_selection(self.NPC, self.N, self.rng)
        self.pop = self.PC

    def _exploration(self):
        return exploration(self, self.PC, self.NPC, self.nND, self.N)

    def step(self):
        rng, W, nr = self.rng, self.W, self.nr
        new_pc = self._exploration()
        n_npc = len(self.NPC)
        if new_pc is not None:
            Fn = objs(new_pc)
            for i in range(len(new_pc)):
                self.Z = np.minimum(self.Z, Fn[i])
                P = rng.permutation(n_npc)
                g_old = np.max(np.abs(objs(self.NPC[P]) - self.Z) / W[P], axis=1)
                g_new = np.max(np.abs(Fn[i] - self.Z) / W[P], axis=1)
                hit = np.where(g_old >= g_new)[0]
                if len(hit):
                    self.NPC[P[hit[:1]]] = new_pc[i]
        new_npc = []
        for i in range(n_npc):
            P = self.B[i][rng.permutation(self.T)] if rng.random() < 0.9 else rng.permutation(n_npc)
            child = self.evaluate(ga_half(self.problem, decs(self.NPC[P[:2]]), rng=rng))
            new_npc.append(child[0])
            fc = objs(child)[0]
            self.Z = np.minimum(self.Z, fc)
            g_old = np.max(np.abs(objs(self.NPC[P]) - self.Z) / W[P], axis=1)
            g_new = np.max(np.abs(fc - self.Z) / W[P], axis=1)
            hit = np.where(g_old >= g_new)[0][:nr]
            self.NPC[P[hit]] = child[0]
        parts = [self.PC, Population.create(new_npc)] + ([new_pc] if new_pc is not None else [])
        merged = parts[0]
        for p in parts[1:]:
            merged = Population.merge(merged, p)
        self.PC, self.nND = _pc_selection(merged, self.N, rng)
        self.pop = self.PC
