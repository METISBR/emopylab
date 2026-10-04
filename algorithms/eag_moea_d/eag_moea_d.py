# emopylab 2026
"""EAG-MOEA-D (external archive guided MOEA/D).

Reference:
X. Cai, Y. Li, Z. Fan, and Q. Zhang. An external archive guided multiobjective evolutionary
algorithm based on decomposition for combinatorial optimization. IEEE Transactions on Evolutionary
Computation, 2015, 19(4): 508-523.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, crowding, decs, ga_half, nd_sort, neighbors_of, objs,
                                            roulette, uniform_point)
from core.population import Population

ALGORITHM_FLAGS = {'EAGMOEAD': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _update_archive(arch, off):
    N = len(arch)
    arch = Population.merge(arch, off)
    F = objs(arch)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-crowding(F[last]), kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return arch[nxt], nxt[N:]


class EAGMOEAD(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, LGs: int = 8, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.LGs = int(LGs)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.swarm = self.pop
        self.archive = self.pop
        self.s = np.zeros((self.N, self.LGs))
        self.pop = self.archive

    def step(self):
        N, T, rng = self.N, self.T, self.rng
        S = self.s.sum(axis=1) + 1e-6
        D = S / S.sum() + 0.002
        D = D / D.sum()
        loc = roulette(N, 1.0 / D, rng=rng)
        pool = np.stack([self.B[loc[i], rng.permutation(T)[:2]] for i in range(N)])        # (N, 2)
        parents = np.vstack([decs(self.swarm[pool[:, 0]]), decs(self.swarm[pool[:, 1]])])
        off = self.evaluate(ga_half(self.problem, parents, rng=rng))
        Fo = objs(off)
        for i in range(len(off)):
            nb = self.B[loc[i]]
            g_old = np.sum(objs(self.swarm[nb]) * self.W[nb], axis=1)
            g_new = self.W[nb] @ Fo[i]
            self.swarm[nb[g_old >= g_new]] = off[i]
        self.archive, success = _update_archive(self.archive, off)
        if success.any():
            self.s[:, int(np.ceil(self.FE / N)) % self.LGs] = np.bincount(loc[success], minlength=N)
        self.pop = self.archive
