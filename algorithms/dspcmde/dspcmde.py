# emopylab 2026
"""DSPCMDE (dynamic selection preference-assisted constrained multiobjective differential evolution).

Reference:
K. Yu, J. Liang, B. Qu, Y. Luo, and C. Yue. Dynamic selection preference-assisted constrained
multiobjective differential evolution. IEEE Transactions on Systems, Man, and Cybernetics: Systems,
2022, 52(5): 2954-2965.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, first_front, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'DSPCMDE': {'constrained', 'integer', 'multi', 'real'}}

_FS = (0.6, 0.8, 1.0)
_CRS = (0.1, 0.2, 1.0)


def _pick(rng, values):
    l = rng.random()
    return values[0] if l <= 1 / 3 else (values[1] if l <= 2 / 3 else values[2])


def _rank(pop, constrained):
    F = objs(pop)
    C = cons(pop) if constrained else np.zeros((len(pop), 0))
    front, _ = nd_sort(F, C if C.size else None, np.inf)
    cd = crowding(F, front)
    order = np.lexsort((-cd, front))
    r = np.zeros(len(pop))
    r[order] = np.arange(1, len(pop) + 1)
    return r, front, cd


def environmental_selection(pop, N, a):
    Rc, front1, cd1 = _rank(pop, True)
    Rp, _, _ = _rank(pop, False)
    rank = np.argsort((1 - a) * Rc + a * Rp, kind="stable")[:N]
    return pop[rank]


class DSPCMDE(LoopAlgorithm):
    """Dynamic selection-pressure constrained DE: DE/current-to-pbest-like variation with random F and CR, and a
    survivor selection that blends the constrained rank with the objective-only rank, moving from the constrained
    ranking (start) to the objective ranking (end)."""

    def _de(self):
        pop, rng, N, D = self.pop, self.rng, self.N, self.D
        X = decs(pop)
        C = cons(pop)
        best_idx = np.where(first_front(objs(pop), C if C.size else None))[0]
        best = int(best_idx[int(np.floor(rng.random() * len(best_idx)))])
        lo, up = self.lower, self.upper
        trial = np.zeros((N, D))
        for i in range(N):
            if rng.random() > 0.5:
                F, CR = _pick(rng, _FS), _pick(rng, _CRS)
                idx = list(range(N))
                idx.pop(i)
                r1 = int(np.floor(rng.random() * (N - 1)))
                x1 = idx.pop(r1)
                r2 = int(np.floor(rng.random() * (N - 2)))
                x2 = idx[r2]
                r3 = int(np.floor(rng.random() * (N - 3)))
                x3 = idx[r3]
                v = X[x1] + rng.random() * (X[best] - X[x1]) + F * (X[x2] - X[x3])
                v = np.minimum(np.maximum(v, lo), up)
                site = rng.random(D) < CR
                site[int(np.floor(rng.random() * D))] = True
                trial[i] = np.where(site, v, X[i])
            else:
                F = _pick(rng, _FS)
                idx = list(range(N))
                idx.pop(i)
                r1 = int(np.floor(rng.random() * (N - 1)))
                x1 = idx.pop(r1)
                r2 = int(np.floor(rng.random() * (N - 2)))
                x2 = idx.pop(r2)
                r3 = int(np.floor(rng.random() * (N - 3)))
                x3 = idx[r3]
                v = X[i] + rng.random() * (X[x1] - X[i]) + F * (X[x2] - X[x3])
                trial[i] = np.minimum(np.maximum(v, lo), up)
        return self.evaluate(trial)

    def step(self):
        off = self._de()
        a = 0.5 * (1 - np.cos((1 - self.FE / self.max_FE) * np.pi))
        self.pop = environmental_selection(Population.merge(self.pop, off), self.N, a)
