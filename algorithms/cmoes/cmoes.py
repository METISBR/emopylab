# emopylab 2026
"""CMOES (constrained multi-objective optimization based on even search).

Reference:
F. Ming, W. Gong, and Y. Jin. Even search in a promising region for constrained multi-objective
optimization. IEEE/CAA Journal of Automatica Sinica, 2024, 11(2): 474-486.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness, environmental_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, de, ga, ga_half, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'CMOES': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _cv(pop):
    C = cons(pop)
    return np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))


def nsga2_selection(pop, N):
    C = cons(pop)
    front, maxf = nd_sort(objs(pop), C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(objs(pop), front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


def _labels(S, nd):
    if len(nd) == 0:
        return np.ones(len(S), bool)
    dominated = (S[:, None, :] > nd[None, :, :]).any(axis=2) & ~(S[:, None, :] < nd[None, :, :]).any(axis=2)
    return ~dominated.any(axis=1)


def even_search(pop, non_dom, N, tau, max_cv):
    first = pop[_labels(objs(pop), objs(non_dom))]
    fit = cal_fitness(objs(first))
    nxt = fit < 1
    if nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(first)[nxt], int(nxt.sum()) - N)]] = False
    pop2, fit2 = first[nxt], fit[nxt]
    cv = _cv(first)
    n3 = cv <= max_cv * (1 - tau) ** 2
    if n3.sum() > N:
        idx = np.where(n3)[0]
        n3[idx[_truncation(objs(first)[n3], int(n3.sum()) - N)]] = False
    return pop2, fit2, first[n3], cv[n3]


class CMOES(LoopAlgorithm):
    """Constrained multi-objective evolutionary search in two stages: an unconstrained helper runs alongside the
    main population until feasible solutions exist, then an even-search step maintains a non-dominated helper
    population and an epsilon-relaxed helper population that shrink toward feasibility."""

    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type

    def start(self):
        self.P1 = self.pop
        self.P2 = self.evaluate(self.random_decs(self.N))
        _, self.front, self.crowd = nsga2_selection(self.P1, self.N)
        self.fit2 = cal_fitness(objs(self.P2))
        self.P3 = self.P1
        self.changed = False
        self.max_cv = float(max(_cv(self.P1).max(), _cv(self.P2).max()))

    def _breed(self, pop, fit, n_pool, *extra):
        rng, pr = self.rng, self.problem
        pool = tournament(2, n_pool, *([fit] + list(extra)), rng=rng)
        if self.type == 1:
            return self.evaluate(ga_half(pr, decs(pop[pool]), rng=rng))
        h = n_pool // 2
        return self.evaluate(de(pr, decs(pop), decs(pop[pool[:h]]), decs(pop[pool[h:]]), rng=rng))

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        gen = int(np.ceil(self.FE / N))
        G = int(np.ceil(self.max_FE / N))
        CV1 = _cv(self.P1)
        if not np.any(CV1 == 0) or gen <= 0.2 * G:
            pool = tournament(2, 2 * N, self.front, -self.crowd, rng=rng)
            if self.type == 1:
                off = self.evaluate(ga(pr, decs(self.P1[pool]), rng=rng))
            else:
                off = self.evaluate(de(pr, decs(self.P1), decs(self.P1[pool[:N]]), decs(self.P1[pool[N:]]), rng=rng))
            off2 = self._breed(self.P2, self.fit2, 2 * N)
            self.P1, self.front, self.crowd = nsga2_selection(Population.merge(self.P1, off, off2), N)
            self.P2, self.fit2 = environmental_selection(Population.merge(self.P2, off, off2), N, False)
            self.P3 = self.P2
            self.max_cv = max(self.max_cv, float(_cv(Population.merge(self.P1, self.P2)).max()))
        else:
            tau = gen / G
            allp = Population.merge(self.P1, self.P2, self.P3)
            self.max_cv = float(_cv(allp).max())
            if not self.changed:
                _, self.fit2, _, self.fit3 = even_search(allp, self.P1[CV1 == 0], N, tau, self.max_cv)
                _, self.fit1 = environmental_selection(self.P1, N, True)
                self.changed = True
            off2 = self._breed(self.P2, self.fit2, 2 * len(self.P2)) if len(self.P2) else None
            off3 = self._breed(self.P3, self.fit3, 2 * len(self.P3)) if len(self.P3) else None
            off1 = self._breed(self.P1, self.fit1, 2 * N)
            off = Population.merge(off1, off2, off3)
            merged = Population.merge(allp, off)
            self.P1, self.fit1 = environmental_selection(merged, N, True)
            self.P2, self.fit2, self.P3, self.fit3 = even_search(merged, self.P1[CV1 == 0], N, tau, self.max_cv)
        self.pop = self.P1
