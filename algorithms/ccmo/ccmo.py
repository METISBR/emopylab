# emopylab 2026
"""CCMO (coevolutionary constrained multi-objective optimization framework).

Reference:
Y. Tian, T. Zhang, J. Xiao, X. Zhang, and Y. Jin. A coevolutionary framework for constrained multi-
objective optimization problems. IEEE Transactions on Evolutionary Computation, 2021, 25(1):
102-116.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, ga_half, objs, pdist2, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'CCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def cal_fitness(F, C=None, eps=None):
    """SPEA2-style strength/raw fitness plus k-th nearest neighbour density; constraint violation ranks first
    (violations up to ``eps`` count as feasible)."""
    N = len(F)
    CV = np.zeros(N) if C is None or np.size(C) == 0 else np.sum(np.maximum(0.0, C), axis=1)
    if eps is not None:
        CV = np.where(CV <= eps, 0.0, CV)
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    Dom = (CV[:, None] < CV[None, :]) | ((CV[:, None] == CV[None, :]) & (k == 1))
    S = Dom.sum(axis=1)
    R = S @ Dom
    dist = pdist2(F, F)
    np.fill_diagonal(dist, np.inf)
    dist = np.sort(dist, axis=1)
    D = 1.0 / (dist[:, max(int(np.floor(np.sqrt(N))) - 1, 0)] + 2)
    return R + D


def _truncation(F, K):
    dist = pdist2(F, F)
    np.fill_diagonal(dist, np.inf)
    return truncate_lexi(dist, K)


def environmental_selection(pop, N, is_origin, eps=None):
    F = objs(pop)
    fit = cal_fitness(F, cons(pop), eps) if is_origin else cal_fitness(F)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(F[nxt], int(nxt.sum()) - N)]] = False
    pop, fit = pop[nxt], fit[nxt]
    rank = np.argsort(fit, kind="stable")
    return pop[rank], fit[rank]


class CCMO(LoopAlgorithm):
    """Coevolutionary framework: one population handles the constraints, a helper population ignores them."""

    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type

    def start(self):
        self.pop1 = self.pop
        self.pop2 = self.evaluate(self.random_decs(self.N))
        self.fit1 = cal_fitness(objs(self.pop1), cons(self.pop1))
        self.fit2 = cal_fitness(objs(self.pop2))

    def step(self):
        N, rng, P1, P2 = self.N, self.rng, self.pop1, self.pop2
        if self.type == 1:
            m1, m2 = tournament(2, N, self.fit1, rng=rng), tournament(2, N, self.fit2, rng=rng)
            off1 = self.evaluate(ga_half(self.problem, decs(P1[m1]), rng=rng))
            off2 = self.evaluate(ga_half(self.problem, decs(P2[m2]), rng=rng))
        else:
            m1, m2 = tournament(2, 2 * N, self.fit1, rng=rng), tournament(2, 2 * N, self.fit2, rng=rng)
            off1 = self.evaluate(de(self.problem, decs(P1), decs(P1[m1[:N]]), decs(P1[m1[N:]]), rng=rng))
            off2 = self.evaluate(de(self.problem, decs(P2), decs(P2[m2[:N]]), decs(P2[m2[N:]]), rng=rng))
        self.pop1, self.fit1 = environmental_selection(Population.merge(P1, off1, off2), N, True)
        self.pop2, self.fit2 = environmental_selection(Population.merge(P2, off1, off2), N, False)
        self.pop = self.pop1
