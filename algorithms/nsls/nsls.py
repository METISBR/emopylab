# emopylab 2026
"""NSLS (multiobjective optimization framework based on nondominated sorting and).

Reference:
B. Chen, W. Zeng, Y. Lin, and D. Zhang. A new local search-based multiobjective optimization
algorithm. IEEE Transactions on Evolutionary Computation, 2015, 19(1): 50-73.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, objs, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'NSLS': {'integer', 'multi', 'real'}}


def _last_selection(F, K, rng):
    N = len(F)
    choose = np.zeros(N, bool)
    choose[np.argmin(F, axis=0)] = True
    choose[np.argmax(F, axis=0)] = True
    if choose.sum() > K:
        chosen = np.where(choose)[0]
        choose[chosen[rng.permutation(len(chosen))[: int(choose.sum()) - K]]] = False
    elif choose.sum() < K:
        dist = pdist2(F, F)
        np.fill_diagonal(dist, np.inf)
        while choose.sum() < K:
            remain = np.where(~choose)[0]
            x = np.argmax(dist[np.ix_(~choose, choose)].min(axis=1))
            choose[remain[x]] = True
    return choose


def _environmental_selection(pop, N, rng):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    nxt[last[_last_selection(F[last], N - int(nxt.sum()), rng)]] = True
    return pop[nxt]


class NSLS(LoopAlgorithm):
    """Every solution is refined one variable at a time by a differential move of two random members."""

    def __init__(self, pop_size: int = 100, mu: float = 0.5, delta: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.mu, self.delta = float(mu), float(delta)

    def _operator(self, pop):
        rng = self.rng
        pop = pop.copy(deep=False)
        N, D = len(pop), self.D
        for i in range(N):
            for d in range(D):
                c = self.mu + self.delta * rng.standard_normal()
                k = rng.integers(0, N, size=2)
                w = np.tile(decs(pop[i:i + 1])[0], (2, 1))
                diff = pop[int(k[0])].get("X")[d] - pop[int(k[1])].get("X")[d]
                w[0, d] += c * diff
                w[1, d] -= c * diff
                w = self.evaluate(w)
                fw, fi = objs(w), objs(pop[i:i + 1])[0]
                kk = [int(np.any(fw[j] < fi)) - int(np.any(fw[j] > fi)) for j in range(2)]
                if kk[0] == -1 and kk[1] == -1:
                    continue
                pick = 0 if kk[0] > kk[1] else 1 if kk[0] < kk[1] else int(rng.integers(0, 2))
                pop[i] = w[pick]
        return pop

    def step(self):
        off = self._operator(self.pop)
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N, self.rng)
