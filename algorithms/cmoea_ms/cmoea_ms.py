# emopylab 2026
"""CMOEA-MS (constrained multiobjective evolutionary algorithm with multiple stages).

Reference:
Y. Tian, Y. Zhang, Y. Su, X. Zhang, K. C. Tan, and Y. Jin. Balancing objective optimization and
constraint satisfaction in constrained evolutionary multi-objective optimization. IEEE Transactions
on Cybernetics, 2022, 52(9): 9559-9572.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, ga, objs, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'CMOEAMS': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _cosine_distance(A, B=None):
    """Cosine distance; a zero vector gives NaN (as the reference distance function does)."""
    B = A if B is None else B
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    with np.errstate(all="ignore"):
        return 1.0 - (A @ B.T) / (na[:, None] * nb[None, :])


def cal_cv(C):
    if C.size == 0:
        return np.zeros(len(C))
    C = np.maximum(C, 0)
    with np.errstate(all="ignore"):
        CV = C / C.max(axis=0)
    CV[:, np.isnan(CV[0])] = 0
    return CV.mean(axis=1)


def cal_sde(F):
    N = len(F)
    with np.errstate(all="ignore"):
        F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        dis = np.linalg.norm(np.maximum(F[None, :, :] - F[:, None, :], 0.0), axis=2)
    k = int(np.floor(np.sqrt(N)))
    return 1.0 / (np.sort(dis, axis=1)[:, k] + 2)


def cal_fitness(F, CV=None):
    N = len(F)
    CV = np.zeros(N) if CV is None else CV
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    Dom = (CV[:, None] < CV[None, :]) | ((CV[:, None] == CV[None, :]) & (k == 1))
    R = Dom.sum(axis=1) @ Dom
    dist = _cosine_distance(F)
    np.fill_diagonal(dist, np.inf)
    D = 1.0 / (np.sort(dist, axis=1)[:, max(int(np.floor(np.sqrt(N))) - 1, 0)] + 2)
    return R + D


def _truncation(F, K):
    dist = _cosine_distance(F)
    np.fill_diagonal(dist, np.inf)
    return truncate_lexi(dist, K)


def environmental_selection(fit, pop, N):
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt], fit[nxt]


class CMOEAMS(LoopAlgorithm):
    """Constrained multi-objective EA with multi-stage fitness: shift-based density estimation with the
    constraint violation as an extra objective, switching to plain objectives once the population is feasible."""

    def __init__(self, pop_size: int = 100, sampling=None, type=1, lambda_: float = 0.5, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type, self.lam = type, float(lambda_)

    def start(self):
        F = objs(self.pop)
        self.fit = cal_fitness(np.column_stack([cal_sde(F), cal_cv(cons(self.pop))]))

    def step(self):
        N, rng, pop = self.N, self.rng, self.pop
        if self.type == 1:
            off = self.evaluate(ga(self.problem, decs(pop[tournament(2, N, self.fit, rng=rng)]), rng=rng))
        else:
            m1, m2 = tournament(2, N, self.fit, rng=rng), tournament(2, N, self.fit, rng=rng)
            off = self.evaluate(de(self.problem, decs(pop), decs(pop[m1]), decs(pop[m2]), rng=rng))
        Q = Population.merge(pop, off)
        CV = cal_cv(cons(Q))
        if np.mean(CV <= 0) > self.lam and self.FE >= 0.1 * self.max_FE:
            fit = cal_fitness(objs(Q), CV)
        else:
            fit = cal_fitness(np.column_stack([cal_sde(objs(Q)), CV]))
        self.pop, self.fit = environmental_selection(fit, Q, N)
