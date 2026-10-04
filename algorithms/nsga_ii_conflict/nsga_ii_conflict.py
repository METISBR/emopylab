# emopylab 2026
"""NSGA-II-conflict (nSGA-II with conflict-based partitioning strategy).

Reference:
A. L. Jaimes, C. A. Coello Coello, H. Aguirre, and K. Tanaka. Objective space partitioning using
conflict information for solving many-objective problems. Information Sciences, 2014, 268: 305-327.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'NSGAIIconflict': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def _conflict_partition(F, NS):
    M = F.shape[1]
    with np.errstate(all="ignore"):
        c = 1.0 - np.nan_to_num(np.corrcoef(F, rowvar=False))
    c = np.tile(c.max(axis=1), (M, 1)) * (1 - np.eye(M))
    k = int(np.ceil(M / NS))
    psi, remain = [], np.arange(M)
    for _ in range(NS - 1):
        L = np.argsort(c[np.ix_(remain, remain)], axis=1, kind="stable")
        i = int(np.argmin(L[:, k - 1]))
        psi.append(remain[L[i, :k]])
        remain = np.delete(remain, L[i, :k])
    psi.append(remain)
    return psi


def _sub_selection(F, N):
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    idx = np.where(nxt)[0]
    return idx, front_no[idx], cd[idx]


def _environmental_selection(pop, N, psi):
    F = objs(pop)
    sel, fno, cdis = np.zeros(N, int), np.zeros(N), np.zeros(N)
    step = int(np.ceil(N / len(psi)))
    for i, cols in enumerate(psi):
        index = np.arange(i * step, min(N, (i + 1) * step))
        sel[index], fno[index], cdis[index] = _sub_selection(F[:, cols], len(index))
    return pop[sel], fno, cdis


class NSGAIIconflict(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, NS: int = 2, cycles: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.NS = int(NS)
        self.cycles = int(cycles)

    def start(self):
        self.Gc = int(np.ceil(self.max_FE / self.N / self.cycles))
        self.psi = [np.arange(self.M)]
        self.phase = True
        self.pop, self.front_no, self.crowd = _environmental_selection(self.pop, self.N, self.psi)

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.crowd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.front_no, self.crowd = _environmental_selection(Population.merge(self.pop, off), self.N, self.psi)
        share = (int(np.ceil(self.FE / self.N)) % self.Gc) / self.Gc
        if not self.phase and share < 0.3:
            self.psi, self.phase = [np.arange(self.M)], True
        elif self.phase and share >= 0.3:
            self.psi, self.phase = _conflict_partition(objs(self.pop), self.NS), False
