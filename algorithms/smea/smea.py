# emopylab 2026
"""SMEA (self-organizing multiobjective evolutionary algorithm).

Reference:
H. Zhang, A. Zhou, S. Song, Q. Zhang, X. Gao, and J. Zhang. A self- organizing multiobjective
evolutionary algorithm. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 792-806.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, de, nd_sort, objs
from algorithms.hype.hype import cal_hv
from algorithms.moea_dd.moea_dd import update_front
from core.population import Population

ALGORITHM_FLAGS = {'SMEA': {'integer', 'multi', 'real'}}


def _pdist(A, B):
    return np.sqrt(np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0))


class SMEA(LoopAlgorithm):
    """Steady-state evolution guided by a self-organizing map: the map (trained on the newest decision vectors) places
    every solution on a low-dimensional lattice, mating happens between lattice neighbours and each offspring replaces
    the worst solution of the incrementally maintained non-dominated ranking (crowding on dominated fronts, hypervolume
    contribution on the first one)."""

    def __init__(self, pop_size: int = 100, d=None, tau0: float = 0.7, h: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.grid, self.tau0, self.H = d, float(tau0), int(h)

    def initial_size(self):
        M = self.M
        if self.grid is None:
            self.grid = [int(np.ceil(self.pop_size ** (1 / (M - 1))))] * (M - 1)
        self.pop_size = int(np.prod(self.grid))
        return self.pop_size

    PER_STEP_OPTIMUM = True

    def start(self):
        N, M = self.N, self.M
        D = np.asarray(self.grid, dtype=float)
        self.sigma0 = np.sqrt(np.sum(D ** 2) / (M - 1)) / 2
        F = objs(self.pop)
        self.front, _ = nd_sort(F, None, np.inf)
        self.S = decs(self.pop)
        self.Wn = self.S.copy()
        axes = [np.arange(1, int(g) + 1) for g in self.grid]
        mesh = np.meshgrid(*axes, indexing="ij")
        Z = np.column_stack([m.reshape(-1, order="F") for m in mesh])      # first lattice dimension varies fastest
        self.LDis = _pdist(Z, Z)
        self.B = np.argsort(self.LDis, axis=1, kind="stable")[:, 1: min(self.H + 1, N)]

    def _select(self, pop, front, y):
        pop = Population.merge(pop, y)
        F = objs(pop)
        N, M = F.shape
        front = update_front(F, front)
        if front.max() > 1:
            Dm = _pdist(F, F)
            np.fill_diagonal(Dm, np.inf)
            last = np.where(front == front.max())[0]
            worst = last[int(np.argmin(Dm[last].min(axis=1)))]
        else:
            delta = np.full(N, np.inf)
            if M == 2:
                rank = np.lexsort((F[:, 1], F[:, 0]))
                for i in range(1, N - 1):
                    delta[rank[i]] = (F[rank[i + 1], 0] - F[rank[i], 0]) * (F[rank[i - 1], 1] - F[rank[i], 1])
            elif N > 1:
                delta = cal_hv(F, F.max(axis=0) * 1.1, 1, 10000, self.rng)
            worst = int(np.argmin(delta))
        front = update_front(F, front, worst)
        return pop[np.arange(N) != worst], front

    def step(self):
        rng, N = self.rng, self.N
        pop, W, LDis = self.pop, self.Wn, self.LDis
        for s in range(len(self.S)):
            frac = 1 - (self.FE + s + 1) / self.max_FE
            sigma, tau = self.sigma0 * frac, self.tau0 * frac
            u1 = int(np.argmin(_pdist(self.S[[s]], W)[0]))
            U = LDis[u1] < sigma
            if U.any():
                W[U] = W[U] + tau * np.exp(-LDis[u1, U])[:, None] * (self.S[s] - W[U])
        A_idx, U_idx = list(range(N)), list(range(N))
        XU = np.zeros(N, dtype=int)
        X = decs(pop)
        for _ in range(N):
            x = int(rng.integers(0, len(A_idx)))
            u = int(np.argmin(_pdist(X[[A_idx[x]]], W[U_idx])[0]))
            XU[U_idx[u]] = A_idx[x]
            del A_idx[x], U_idx[u]
        old = decs(pop).copy()
        for u in range(N):
            Q = XU[self.B[u]] if rng.random() < 0.9 else np.arange(N)
            Q = Q[rng.permutation(len(Q))[:2]]
            Xc = decs(pop)
            y = self.evaluate(de(self.problem, Xc[[u]], Xc[[Q[0]]], Xc[[Q[1]]], rng=rng))
            pop, self.front = self._select(pop, self.front, y)
        self.pop = pop
        new = decs(pop)
        seen = {tuple(r) for r in old}
        fresh = np.array([r for r in new if tuple(r) not in seen])
        self.S = np.unique(fresh, axis=0) if len(fresh) else np.zeros((0, self.D))
