# emopylab 2026
"""CMODE-FTR (constrained multiobjective differential evolution based on the fusion of two rankings).

Reference:
Z. Zeng, X. Zhang, and Z. Hong. A constrained multiobjective differential evolution algorithm based
on the fusion of two rankings. Information Sciences, 2023, 647, 119572.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, cv, decs, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'CMODEFTR': {'constrained', 'integer', 'multi', 'real'}}


def _ranking(F, C):
    front_no, _ = nd_sort(F, C, np.inf)
    cd = crowding(F, front_no)
    order = np.lexsort((-cd, front_no))
    rank = np.empty(len(F), dtype=int)
    rank[order] = np.arange(1, len(F) + 1)
    return rank


def _environmental_selection(pop, N, a, fe, max_fe):
    F, C = objs(pop), cons(pop)
    Rc = _ranking(F, C if C.size else None)                               # constraint-domination ranking
    Rp = _ranking(F, None)                                                # Pareto ranking
    pro_l = 1.0 - np.sum(cv(pop) > 0) / len(pop)
    b = 1.0 / ((fe / max_fe) ** 2 + 1) - 0.5
    r = a * (1 - b) + pro_l * b
    rank = np.argsort((1 - r) * Rc + r * Rp, kind="stable")
    return pop[rank[:N]]


class CMODEFTR(LoopAlgorithm):
    def _generate(self):
        rng, N, D = self.rng, self.N, self.D
        X = decs(self.pop)
        F, C = objs(self.pop), cons(self.pop)
        front_no, _ = nd_sort(F, C if C.size else None, 1)
        nd = np.where(front_no == 1)[0]
        lo, up = self.lower, self.upper
        trial = np.zeros((N, D))
        for i in range(N):
            best = int(nd[int(np.floor(rng.random() * len(nd)))])
            idx = list(range(N))
            idx.pop(i)
            xr1 = idx.pop(int(np.floor(rng.random() * (N - 1))))
            xr2 = idx.pop(int(np.floor(rng.random() * (N - 2))))
            xr3 = idx[int(np.floor(rng.random() * (N - 3)))]
            if rng.random() <= 0.5:
                CR = 0.1
                if rng.random() < 0.5:
                    Fs = rng.choice([0.6, 0.8, 1.0])
                    v = X[xr1] + rng.random() * (X[best] - X[xr1]) + Fs * (X[xr2] - X[xr3])
                else:
                    Fs = rng.choice([0.1, 0.8, 1.0])
                    v = X[xr1] + Fs * (X[i] - X[xr1]) + Fs * (X[xr2] - X[xr3])
                v = np.minimum(np.maximum(v, lo), up)
                site = rng.random(D) < CR
                site[int(np.floor(rng.random() * D))] = True
                trial[i] = np.where(site, v, X[i])
            else:
                if rng.random() < 0.5:
                    Fs = rng.choice([0.6, 0.8, 1.0])
                    v = X[i] + rng.random() * (X[xr1] - X[i]) + Fs * (X[xr2] - X[xr3])
                else:
                    Fs = rng.choice([0.1, 0.8, 1.0])
                    v = X[i] + Fs * (X[best] - X[i]) + Fs * (X[xr1] - X[xr2])
                trial[i] = np.minimum(np.maximum(v, lo), up)
        return self.evaluate(trial)

    def step(self):
        off = self._generate()
        t, max_gen = self.FE / self.N, self.max_FE / self.N
        a = 0.5 * (1 - np.cos((1 - t / max_gen) * np.pi))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N, a, self.FE, self.max_FE)
