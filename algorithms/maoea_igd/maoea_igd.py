# emopylab 2026
"""MaOEA-IGD (iGD based many-objective evolutionary algorithm).

Reference:
Y. Sun, G. G. Yen, and Z. Yi. IGD indicator-based evolutionary algorithm for many-objective
optimization problems. IEEE Transactions on Evolutionary Computation, 2019, 23(2): 173-187.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, ga_half, objs, tournament, truncate_lexi, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MaOEAIGD': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def _fitness(F):
    fit = np.zeros_like(F)
    for i in range(F.shape[1]):
        rest = np.delete(F, i, axis=1)
        fit[:, i] = np.abs(F[:, i]) + 100 * np.sum(rest ** 2, axis=1)
    return fit


def _env_selection(pop, W, N):
    """Three-level selection against the reference points ``W`` (dominating < incomparable < dominated), the last level
    being filled by assigning one solution to every remaining well-spread reference point."""
    _, x = np.unique(np.round(objs(pop) * 1e4) / 1e4, axis=0, return_index=True)
    pop = pop[x]
    F = objs(pop)
    N = min(N, len(pop))
    n, NW = len(pop), len(W)
    rank = np.zeros(n, int)
    dis = np.zeros((n, NW))
    lt = (F[:, None, :] < W[None]).any(axis=2)
    gt = (F[:, None, :] > W[None]).any(axis=2)
    domi = lt.astype(int) - gt.astype(int)
    d = np.sqrt(np.sum((F[:, None, :] - W[None]) ** 2, axis=2))
    dd = np.sqrt(np.sum(np.maximum(F[:, None, :] - W[None], 0) ** 2, axis=2))
    for i in range(n):
        if np.any(domi[i] == 1):
            rank[i], dis[i] = 1, -d[i]
        elif np.any(domi[i] == -1):
            rank[i], dis[i] = 3, d[i]
        else:
            rank[i], dis[i] = 2, dd[i]
    cum = np.cumsum([np.sum(rank == k) for k in (1, 2, 3)])
    max_f = int(np.argmax(cum >= N)) + 1
    nxt = rank < max_f
    last = np.where(rank == max_f)[0]
    K = N - int(nxt.sum())
    D = np.sqrt(np.sum((W[:, None, :] - W[None]) ** 2, axis=2))
    np.fill_diagonal(D, np.inf)
    del_ = truncate_lexi(D, max(NW - K, 0)) if NW > K else np.zeros(NW, bool)
    Dl = dis[last][:, ~del_]
    choose = np.zeros(len(last), bool)
    for i in range(Dl.shape[1]):
        remain = np.where(~choose)[0]
        if len(remain) == 0:
            break
        choose[remain[int(np.argmin(Dl[remain, i]))]] = True
    nxt[last[choose]] = True
    return pop[nxt], rank[nxt], dis[nxt]


class MaOEAIGD(LoopAlgorithm):
    """Stage 1 (up to ``DNPE`` evaluations) finds the extreme points that scale a set of uniform reference points;
    stage 2 restarts from a fresh population and selects it by its position relative to those reference points."""

    def __init__(self, pop_size: int = 100, dnpe=None, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.dnpe = dnpe

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        self.DNPE = 100 * self.N if self.dnpe is None else float(self.dnpe)
        self.phase = 1

    def _stage1(self):
        rng, N, M = self.rng, self.N, self.M
        pop = self.pop
        X = decs(pop)
        off = self.evaluate(ga(self.problem, X[rng.integers(0, len(pop), N)], [0.9, 20, 1, 20], rng=rng))
        pop = Population.merge(pop, off)
        rank = np.argsort(_fitness(objs(pop)), axis=0, kind="stable")
        self.pop = pop[np.unique(rank[: int(np.ceil(N / M))].ravel())]
        if self.FE >= self.max_FE or self.FE < self.DNPE:
            return
        F = objs(self.pop)
        ext = np.argmin(_fitness(F), axis=0)
        zmax = F[ext, np.arange(M)].copy()
        zmin = F.min(axis=0)
        zmax[zmax < 1e-6] = 1
        self.W = self.W * (zmax - zmin) + zmin
        self.phase = 2
        fresh = self.evaluate(self.random_decs(N))
        self.pop, self.rank, self.dis = _env_selection(fresh, self.W, N)

    def step(self):
        if self.phase == 1:
            return self._stage1()
        pop, N = self.pop, self.N
        mate = tournament(2, N, self.rank, self.dis.min(axis=1), rng=self.rng)
        off = self.evaluate(ga_half(self.problem, decs(pop[mate]), rng=self.rng))
        self.pop, self.rank, self.dis = _env_selection(Population.merge(pop, off), self.W, N)
