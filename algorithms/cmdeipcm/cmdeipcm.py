# emopylab 2026
"""CMDEIPCM (constrained multiobjective differential evolution algorithm with an infeasible proportion control mechanism).

Reference:
J. Liang, X. Ban, K. Yu, K. Qiao, and B. Qu. Constrained multiobjective differential evolution
algorithm with infeasible-proportion control mechanism. Knowledge-Based Systems, 2022, 250: 109105.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, objs
from algorithms.community_utils.spea import cal_fitness, overall_cv, truncation
from core.population import Population

ALGORITHM_FLAGS = {'CMDEIPCM': {'constrained', 'integer', 'large', 'multi', 'real'}}


def _select(pop, N, F, C):
    """Non-dominated first, then best fitness ranks / nearest-neighbour truncation; result ordered by fitness."""
    fit = cal_fitness(F, C)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    pop, fit = pop[nxt], fit[nxt]
    order = np.argsort(fit, kind="stable")
    return pop[order], fit[order]


class CMDEIPCM(LoopAlgorithm):
    """Two DE-driven populations exchange donors: one respects the constraints, the other ignores them except that its
    share of infeasible solutions is steered along a cosine schedule (the surplus of infeasible ones is penalised)."""

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(self.pop_size))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop1 = infills
        self.pop2 = self.evaluate(self.random_decs(self.pop_size))
        self.fit1 = cal_fitness(objs(self.pop1), cons(self.pop1))
        self.fit2 = cal_fitness(objs(self.pop2))
        self.pop = self.pop1
        self._set_optimum()

    def _de_generator(self, P1, P2, fit1):
        rng, lo, up = self.rng, self.lower, self.upper
        p1, p2 = decs(P1), decs(P2)
        n, D = p1.shape
        trial = np.zeros((n, D))
        fr = np.where(fit1 < 1)[0]
        for i in range(n):
            F = (0.6, 0.8, 1.0)[int(np.searchsorted([1 / 3, 2 / 3], rng.random(), side="left"))]
            CR = (0.1, 0.2, 1.0)[int(np.searchsorted([1 / 3, 2 / 3], rng.random(), side="left"))]
            idx = [k for k in range(n) if k != i]
            xr1 = idx.pop(int(rng.random() * (n - 1)))
            xr2 = idx.pop(int(rng.random() * (n - 2)))
            xr3 = idx[int(rng.random() * (n - 3))]
            best1 = fr[int(rng.random() * len(fr))]
            if rng.random() < 0.5:
                v = p1[i] + F * (p2[xr1] - p1[i]) + F * (p1[xr2] - p1[xr3])
            else:
                v = p1[i] + F * (p1[best1] - p1[i]) + F * (p1[xr2] - p1[xr3])
            w = np.where(v < lo)[0]
            if len(w):
                v[w] = 2 * lo[w] - v[w]
                w1 = w[v[w] > up[w]]
                v[w1] = up[w1]
            y = np.where(v > up)[0]
            if len(y):
                v[y] = 2 * up[y] - v[y]
                y1 = y[v[y] < lo[y]]
                v[y1] = lo[y1]
            t = rng.random(D) < CR
            t[int(rng.random() * D)] = True
            trial[i] = np.where(t, v, p1[i])
        return self.evaluate(trial)

    def step(self):
        rng, N = self.rng, self.N
        off1 = self._de_generator(self.pop1, self.pop2, self.fit1)
        off2 = self._de_generator(self.pop2, self.pop1, self.fit2)
        allp = Population.merge(self.pop1, off1, off2)
        self.pop1, self.fit1 = _select(allp, N, objs(allp), cons(allp))
        pinfea = 0.5 * (1 - np.cos((1 - self.FE / self.max_FE) * np.pi))
        allp = Population.merge(self.pop2, off1, off2)
        Obj = objs(allp).copy()
        cv = overall_cv(cons(allp))
        infea = np.where(cv > 0)[0]
        p_in = len(infea) / len(allp)
        if p_in > pinfea:
            sec = rng.permutation(len(infea))[: int(np.floor(len(allp) * (p_in - pinfea)))]
            Obj[infea[sec]] += Obj.max(axis=0)
        self.pop2, self.fit2 = _select(allp, N, Obj, None)
        self.pop = self.pop1
