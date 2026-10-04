# emopylab 2026
"""CMMO (coevolutionary multi-modal multi-objective optimization framework).

Reference:
F. Ming, W. Gong, L. Wang, and L. Gao. Balancing convergence and diversity in objective and decision
spaces for multimodal multi-objective optimization. IEEE Transactions on Emerging Topics in
Computational Intelligence, 2023, 7(2): 474-486.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga_half, objs, pdist2, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'CMMO': {'binary', 'integer', 'label', 'multi', 'multimodal', 'permutation', 'real'}}


def _density(X, N):
    dist = pdist2(X, X)
    np.fill_diagonal(dist, np.inf)
    return 1.0 / (np.sort(dist, axis=1)[:, max(int(np.floor(np.sqrt(N))) - 1, 0)] + 2)


def _raw(Dom):
    return Dom.sum(axis=1) @ Dom


def cal_fitness(F, X):
    N = len(F)
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    R = _raw(k == 1)
    D_pop, D_dec = _density(F, N), _density(X, N)
    return D_dec, D_pop, R + D_pop + D_dec


def cal_fitness_dec_epsilon(F, X, eps):
    N = len(F)
    d = F[None, :, :] - F[:, None, :]                           # d[i, j] = F_j - F_i
    k = (d > eps).any(axis=2).astype(int) - (d < eps).any(axis=2).astype(int)
    Dom = np.triu(k == 1, 1) | np.triu(k == -1, 1).T
    D = _density(X, N)
    return _raw(Dom) + D, D


def _truncate(D, K):
    np.fill_diagonal(D, np.inf)
    return truncate_lexi(D, K)


def env_selection(pop, N):
    F, X = objs(pop), decs(pop)
    D_dec, D_pop, fit = cal_fitness(F, X)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        dm = pdist2(F[nxt], F[nxt]) + pdist2(X[nxt], X[nxt])
        nxt[idx[_truncate(dm, int(nxt.sum()) - N)]] = False
    pop, fit, D_dec, D_pop = pop[nxt], fit[nxt], D_dec[nxt], D_pop[nxt]
    r = np.argsort(fit, kind="stable")
    return pop[r], fit[r], D_dec[r], D_pop[r]


def env_selection_dec(pop, N, eps):
    F, X = objs(pop), decs(pop)
    fit, D = cal_fitness_dec_epsilon(F, X, eps)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncate(pdist2(X[nxt], X[nxt]), int(nxt.sum()) - N)]] = False
    pop, fit, D = pop[nxt], fit[nxt], D[nxt]
    r = np.argsort(fit, kind="stable")
    return pop[r], fit[r], D[r]


class CMMO(LoopAlgorithm):
    """Constrained-style multimodal EA: one population ranks by objectives and decision-space diversity, the other
    by an epsilon-relaxed dominance that is tightened over the run, exchanging offspring between them."""

    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type

    def start(self):
        N = self.N
        self.P1 = self.pop
        self.P2 = self.evaluate(self.random_decs(N))
        self.G = int(np.ceil(self.max_FE / (2 * N)))
        self.gen, self.Tc, self.tao, self.thr = 1, 0.8 * self.G, 0.1, 0.1
        self.eps0 = float(np.sum(objs(self.P2).max(axis=0)))
        self.eps_k = self.eps0
        self.D_dec, self.D_pop, self.fit1 = cal_fitness(objs(self.P1), decs(self.P1))
        self.fit2, self.Ddec2 = cal_fitness_dec_epsilon(objs(self.P2), decs(self.P2), self.eps_k)

    def step(self):
        N, rng, G, gen = self.N, self.rng, self.G, self.gen
        if gen < 0.2 * G:
            self.eps_k = 0
        elif gen < self.Tc:
            self.eps_k = (1 - self.tao) * self.eps_k
            if self.eps_k <= self.thr:
                self.eps_k = (gen / G) * self.eps0
        m1 = tournament(2, N, self.D_dec, self.D_pop, self.fit1, rng=rng)
        m2 = tournament(2, N, self.Ddec2, self.fit2, rng=rng)
        off1 = self.evaluate(ga_half(self.problem, decs(self.P1[m1]), rng=rng))
        off2 = self.evaluate(ga_half(self.problem, decs(self.P2[m2]), rng=rng))
        self.P1, self.fit1, self.D_dec, self.D_pop = env_selection(Population.merge(self.P1, off1, off2), N)
        self.P2, self.fit2, self.Ddec2 = env_selection_dec(Population.merge(self.P2, off1, off2), N, self.eps_k)
        self.gen = int(np.ceil(self.FE / (2 * N)))
        self.eps0 = min(float(np.sum(objs(self.P2), axis=1).max()), self.eps0)
        self.pop = self.P1
