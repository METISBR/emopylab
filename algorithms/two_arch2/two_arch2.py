# emopylab 2026
"""Two_Arch2 (two-archive algorithm 2).

Reference:
H. Wang, L. Jiao, and X. Yao. Two_Arch2: An improved two-archive algorithm for many-objective
optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 524-541.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, first_front, ga, objs
from algorithms.ibea.ibea import truncate_by_fitness
from core.population import Population

ALGORITHM_FLAGS = {'Two_Arch2': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _update_ca(CA, new, max_size):
    CA = new if CA is None else Population.merge(CA, new)
    if len(CA) <= max_size:
        return CA
    return truncate_by_fitness(CA, max_size, 0.05)


def _update_da(DA, new, max_size, p, rng):
    DA = new if DA is None else Population.merge(DA, new)
    DA = DA[first_front(objs(DA))]
    N = len(DA)
    if N <= max_size:
        return DA
    F = objs(DA)
    choose = np.zeros(N, bool)
    choose[np.argmin(F, axis=0)] = True
    choose[np.argmax(F, axis=0)] = True
    if choose.sum() > max_size:
        ch = np.where(choose)[0]
        choose[ch[rng.permutation(len(ch))[: int(choose.sum()) - max_size]]] = False
    elif choose.sum() < max_size:
        dist = np.sum(np.abs(F[:, None, :] - F[None, :, :]) ** p, axis=2) ** (1.0 / p)
        np.fill_diagonal(dist, np.inf)
        while choose.sum() < max_size:
            remain = np.where(~choose)[0]
            x = int(np.argmax(dist[np.ix_(~choose, choose)].min(axis=1)))
            choose[remain[x]] = True
    return DA[choose]


class Two_Arch2(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, CAsize: int | None = None, p: float | None = None, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.CAsize, self.p = CAsize, p

    def start(self):
        self.ca_size = int(self.CAsize) if self.CAsize else self.N
        self.pnorm = float(self.p) if self.p else 1.0 / self.M
        self.CA = _update_ca(None, self.pop, self.ca_size)
        self.DA = _update_da(None, self.pop, self.N, self.pnorm, self.rng)
        self.pop = self.DA

    def step(self):
        rng, N = self.rng, self.N
        h = int(np.ceil(N / 2))
        a, b = rng.integers(0, len(self.CA), size=h), rng.integers(0, len(self.CA), size=h)
        Fa, Fb = objs(self.CA[a]), objs(self.CA[b])
        dom = np.any(Fa < Fb, axis=1).astype(int) - np.any(Fa > Fb, axis=1).astype(int)
        parent_c = Population.merge(self.CA[np.concatenate([a[dom == 1], b[dom != 1]])], self.DA[rng.integers(0, len(self.DA), size=h)])
        parent_m = self.CA[rng.integers(0, len(self.CA), size=N)]
        off = np.vstack([ga(self.problem, decs(parent_c), (1, 20, 0, 0), rng=rng), ga(self.problem, decs(parent_m), (0, 0, 1, 20), rng=rng)])
        off = self.evaluate(off)
        self.CA = _update_ca(self.CA, off, self.ca_size)
        self.DA = _update_da(self.DA, off, N, self.pnorm, rng)
        self.pop = self.DA
