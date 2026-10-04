# emopylab 2026
"""MyO-DEMR (many-objective differential evolution with mutation restriction).

Reference:
R. Denysiuk, L. Costa, and I. E. Santo. Many-objective optimization using differential evolution
with variable-wise mutation restriction. Proceedings of the Annual Conference on Genetic and
Evolutionary Computation, 2013, 591-598.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, objs, pdist2, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MyODEMR': {'integer', 'many', 'multi', 'real'}}


def _truncation(F, P, remaining, rng):
    N1, N2 = len(F), len(P)
    rng_ = F.max(axis=0) - F.min(axis=0)
    rng_[rng_ == 0] = 1.0
    F = (F - F.min(axis=0)) / rng_
    R = P + min(np.min(F.sum(axis=1) - 1.0), 0.0)
    dist = pdist2(R, F)
    order = rng.permutation(N2)
    nxt = []
    for i in range(1, remaining + 1):
        p = (i - 1) % N2
        for idx in np.argsort(dist[order[p]], kind="stable"):
            if idx not in nxt:
                nxt.append(int(idx))
                break
    return np.asarray(nxt, dtype=int)


def _environmental_selection(pop, N, P, rng):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    nxt[last[_truncation(F[last], P, N - int(nxt.sum()), rng)]] = True
    return pop[nxt]


class MyODEMR(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, nP: int = 500, CR: float = 0.15, proM: float = 1.0, disM: float = 20.0, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.nP, self.CR, self.proM, self.disM = int(nP), float(CR), float(proM), float(disM)

    def start(self):
        self.P, _ = uniform_point(self.nP, self.M)

    def _operator(self, P1, P2, P3):
        rng = self.rng
        N, D = P1.shape
        lo, up = self.lower, self.upper
        V = P2 - P3
        span = np.broadcast_to(up - lo, (N, D))
        site = rng.random((N, D)) < self.proM / D
        mu = rng.random((N, D))
        t = site & (mu <= 0.5)
        V[t] = V[t] + span[t] * ((2 * mu[t]) ** (1 / (self.disM + 1)) - 1)
        t = site & (mu > 0.5)
        V[t] = V[t] + span[t] * (1 - (2 * (1 - mu[t])) ** (1 / (self.disM + 1)))
        V = np.minimum(np.maximum(V, (lo - up) / 2), (up - lo) / 2)
        s2 = rng.random((N, D)) < self.CR
        off = P1.copy()
        off[s2] = off[s2] + V[s2]
        return self.evaluate(off)

    def step(self):
        N, rng = self.N, self.rng
        X = decs(self.pop)
        off = self._operator(X[:N], decs(self.pop[rng.integers(0, N, size=N)]), decs(self.pop[rng.integers(0, N, size=N)]))
        self.pop = _environmental_selection(Population.merge(self.pop, off), N, self.P, rng)
