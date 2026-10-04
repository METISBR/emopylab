# emopylab 2026
"""LSMOF (large-scale multi-objective optimization framework with NSGA-II).

Reference:
C. He, L. Li, Y. Tian, X. Zhang, R. Cheng, Y. Jin, and X. Yao. Accelerating large-scale multi-
objective optimization via problem reformulation. IEEE Transactions on Evolutionary Computation,
2019, 23(6): 949-961.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, de, ga, nd_sort, objs, tournament
from algorithms.sgea.sgea import _best
from core.population import Population
from util.hv import hypervolume

ALGORITHM_FLAGS = {'LSMOF': {'integer', 'large', 'multi', 'real'}}


def _selection(pop, N):
    front, maxf = nd_sort(objs(pop), None, N)
    nxt = front < maxf
    cd = crowding(objs(pop), front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


def _hv_score(pop, reference):
    """Hypervolume of the feasible non-dominated solutions in the box [min(f, 0), reference] scaled by 1.1."""
    F = objs(_best(pop))
    M = F.shape[1]
    fmin = np.minimum(F.min(axis=0), np.zeros(M))
    with np.errstate(all="ignore"):
        F = (F - fmin) / ((reference - fmin) * 1.1)
    F = F[~np.any(F > 1, axis=1)]
    return 0.0 if len(F) == 0 else float(hypervolume(F, np.ones(M)))


class LSMOF(LoopAlgorithm):
    """Large-scale multi-objective optimization framework: during the first 60% of the budget the search runs in a
    weight space (one weight per direction from a reference solution to the bounds) optimised by DE against the
    hypervolume of the projected solutions; afterwards a plain NSGA-II variant refines the decision vectors."""

    def __init__(self, pop_size: int = 100, wD: int = 10, SubN: int = 30, Operator: int = 2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wD, self.SubN, self.op = int(wD), int(SubN), int(Operator)

    def start(self):
        self.G = int(np.ceil(self.max_FE * 0.05 / (self.SubN * 2 * self.wD)))

    def _fit(self, w, direct, reference):
        wD, lo, up = self.wD, self.lower, self.upper
        dec = np.vstack([w[:wD, None] * direct[:wD] + lo, up - w[wD:, None] * direct[wD:]])
        off = self.evaluate(dec)
        return -_hv_score(off, reference), off

    def _weight_optimization(self, pop):
        rng, wD, N2, lo, up = self.rng, self.wD, self.SubN, self.lower, self.upper
        reference = objs(pop).max(axis=0)
        ref_pop, _, _ = _selection(pop, wD)
        X = decs(ref_pop)
        direction = np.concatenate([np.sqrt(np.sum((X - lo) ** 2, axis=1)), np.sqrt(np.sum((up - X) ** 2, axis=1))])
        direct = np.vstack([X - lo, up - X]) / direction[:, None]
        wmax = np.sqrt(np.sum((up - lo) ** 2)) * 0.5
        w0 = rng.random((N2, 2 * wD)) * wmax
        fit, new = [], []
        for i in range(N2):
            f, o = self._fit(w0[i], direct, reference)
            fit.append(f)
            new.append(o)
        popnew = Population.merge(*new)
        arc = popnew[nd_sort(objs(popnew), None, 1)[0] == 1]
        pos, cost = w0.copy(), np.array(fit)
        temp = None
        pCR, bmin, bmax = 0.2, 0.2, 0.8
        for _ in range(self.G):
            for i in range(N2):
                A = rng.permutation(N2)
                A = A[A != i]
                a, b, c = A[0], A[1], A[2]
                beta = rng.uniform(bmin, bmax, 2 * wD)
                y = np.clip(pos[a] + beta * (pos[b] - pos[c]), 0, wmax)
                j0 = int(rng.integers(0, 2 * wD))
                cross = rng.random(2 * wD) <= pCR
                cross[j0] = True
                z = np.where(cross, y, pos[i])
                f, o = self._fit(z, direct, reference)
                temp = Population.merge(temp, o)
                temp = temp[nd_sort(objs(temp), None, 1)[0] == 1]
                if f < cost[i]:
                    pos[i], cost[i] = z, f
        arc = Population.merge(arc, temp) if temp is not None else arc
        if len(arc) > self.N:
            arc = arc[nd_sort(objs(arc), None, 1)[0] == 1]
        return arc

    def _sub_nsga2(self, pop):
        N, rng, pr = self.N, self.rng, self.problem
        front = nd_sort(objs(pop), None, np.inf)[0]
        cd = crowding(objs(pop), front)
        if self.op == 1:
            off = self.evaluate(ga(pr, decs(pop[tournament(2, N, front, -cd, rng=rng)]), rng=rng))
        else:
            m1, m2, m3 = (tournament(2, N, front, -cd, rng=rng) for _ in range(3))
            off = self.evaluate(de(pr, decs(pop[m1]), decs(pop[m2]), decs(pop[m3]), rng=rng))
        return _selection(Population.merge(pop, off), N)[0]

    def step(self):
        if self.FE < 0.6 * self.max_FE:
            arc = self._weight_optimization(self.pop)
            self.pop = _selection(Population.merge(self.pop, arc), self.N)[0]
        else:
            self.pop = self._sub_nsga2(self.pop)
