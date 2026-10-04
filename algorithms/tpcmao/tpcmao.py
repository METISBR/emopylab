# emopylab 2026
"""TPCMaO (three-population based constrained many-objective co-evolutionary algorithm).

Reference:
Y. Tian, Z. Shi, Y. Zhang, L. Zhang, H. Zhang, and X. Zhang. Solving optimal power flow problems via
a constrained many-objective co-evolutionary algorithm. Frontiers in Energy Research, 2023, 11:
1293193.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, ga_half, objs, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'TPCMaO': {'binary', 'constrained', 'integer', 'label', 'many', 'permutation', 'real'}}


def _sde_dist(F):
    d = np.linalg.norm(np.maximum(F[None, :, :] - F[:, None, :], 0.0), axis=2)
    np.fill_diagonal(d, np.inf)
    return d


def cal_fitness(F, C=None):
    N = len(F)
    CV = np.zeros(N) if C is None or np.size(C) == 0 else np.sum(np.maximum(0, C), axis=1)
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    Dom = (CV[:, None] < CV[None, :]) | ((CV[:, None] == CV[None, :]) & (k == 1))
    R = Dom.sum(axis=1) @ Dom
    D = 1.0 / (np.sort(_sde_dist(F), axis=1)[:, max(int(np.floor(np.sqrt(N))) - 1, 0)] + 2)
    return R + D


def _truncate(F, K):
    return truncate_lexi(_sde_dist(F), K)


def _survive(pop, fit, N):
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncate(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return nxt


def archive(pop, N):
    C = cons(pop)
    feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
    pop = pop[feas]
    if len(pop) > N:
        pop = pop[_survive(pop, cal_fitness(objs(pop), cons(pop)), N)]
    return pop


def env_selection(pop, N, kind, eps=None):
    F, C = objs(pop), cons(pop)
    if kind == 1:
        fit = cal_fitness(F, C)
    elif kind == 2:
        fit = cal_fitness(F)
    else:
        C = C.copy() if C.size else C
        if C.size:
            C[np.sum(np.maximum(0, C), axis=1) <= eps] = 0
        fit = cal_fitness(F, C)
    fit = fit.copy()
    fit[np.unique(np.argmin(F, axis=0))] = 0
    nxt = _survive(pop, fit, N)
    pop, fit = pop[nxt], fit[nxt]
    Cp = cons(pop)
    cv = np.sum(np.maximum(0, Cp), axis=1) if Cp.size else np.zeros(len(pop))
    return pop, fit, float(np.sum(cv == 0) / len(pop))


class TPCMaO(LoopAlgorithm):
    """Three-population constrained many-objective EA: a constrained population, an unconstrained helper and an
    epsilon-relaxed population whose feasible solutions feed an external archive."""

    def start(self):
        N = self.N
        self.P1 = self.pop
        self.P2 = self.evaluate(self.random_decs(N))
        self.P3 = self.evaluate(self.random_decs(N))
        self.f1 = cal_fitness(objs(self.P1), cons(self.P1))
        self.f2 = cal_fitness(objs(self.P2))
        self.f3 = cal_fitness(objs(self.P3))
        C = cons(self.P3)
        e0 = float(np.max(np.sum(np.maximum(C, 0), axis=1))) if C.size else 0.0
        self.eps0 = e0 if e0 != 0 else 1.0
        self.eps = self.eps0
        self.arch = archive(self.P3, N)

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        off1 = self.evaluate(ga_half(pr, decs(self.P1[tournament(2, N, self.f1, rng=rng)]), rng=rng))
        off2 = self.evaluate(ga_half(pr, decs(self.P2[tournament(2, N, self.f2, rng=rng)]), rng=rng))
        tr, _, fr = env_selection(Population.merge(self.P2, off2), N, 1)
        if fr > 0.5:
            pick = rng.permutation(N)[: N // 2]
            self.P1, self.f1, _ = env_selection(Population.merge(self.P1, off1, tr[pick]), N, 1)
        else:
            self.P1, self.f1, _ = env_selection(Population.merge(self.P1, off1, off2), N, 1)
        self.P2, self.f2, _ = env_selection(Population.merge(self.P2, off2, off1), N, 2)
        C = cons(self.P3)
        cv3 = np.sum(np.maximum(C, 0), axis=1) if C.size else np.zeros(len(self.P3))
        if self.FE < 0.9 * self.max_FE:
            if np.mean(cv3 <= 0.02) < 0.9:
                self.eps = 0.9 * self.eps
            else:
                self.eps = self.eps0 * (1 - self.FE / 0.9 / self.max_FE) ** 2
        else:
            self.eps = 0.0
        off = self.evaluate(ga(pr, decs(self.P3[tournament(2, N, self.f3, rng=rng)]), rng=rng))
        self.P3, self.f3, _ = env_selection(Population.merge(self.P3, off), N, 3, self.eps)
        self.arch = archive(Population.merge(self.arch, self.P3), N)
        if self.FE >= self.max_FE:
            fin = archive(Population.merge(self.arch, self.P1), N)
            if len(fin):
                self.P1 = fin
        self.pop = self.P1
