# emopylab 2026
"""MMOPSO (mOPSO with multiple search strategies).

Reference:
Q. Lin, J. Li, Z. Du, J. Chen, and Z. Ming. A novel multi-objective particle swarm optimization with
multiple search strategies. European Journal of Operational Research, 2015, 247(3): 732-744.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, crowding, decs, first_front, ga_half, objs, uniform_point,
                                            velocity)
from core.population import Population

ALGORITHM_FLAGS = {'MMOPSO': {'integer', 'multi', 'real'}}


def _update_archive(A, N):
    A = A[first_front(objs(A))]
    rank = np.argsort(-crowding(objs(A)), kind="stable")
    return A[rank[: min(N, len(A))]]


class MMOPSO(LoopAlgorithm):
    def initial_size(self):
        W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        return self.pop_size

    def _classification(self, pop):
        F = objs(pop) - self.Z
        P = np.argmax(F @ self.W.T, axis=1)
        nxt = []
        for i in range(len(self.W)):
            cur = np.where(P == i)[0]
            if len(cur) == 0:
                nxt.append(self.evaluate(self.random_decs(1))[0])
            else:
                nd = cur[first_front(objs(pop[cur]))]
                Fn = F[nd]
                best = int(np.argmax((Fn @ self.W[i]) / np.sum(Fn ** 2, axis=1) ** 0.6))
                nxt.append(pop[nd[best]])
        return Population.create(nxt)

    def start(self):
        self.Z = objs(self.pop).min(axis=0)
        self.swarm = self._classification(self.pop)
        self.archive = _update_archive(self.swarm, self.N)
        self.pop = self.archive

    def _get_best(self):
        A = objs(self.archive) - self.Z
        normW = np.linalg.norm(self.W, axis=1)
        d1 = A @ self.W.T / normW
        d2 = np.sqrt(np.maximum(np.sum(A ** 2, axis=1)[:, None] - d1 ** 2, 0.0))
        pbest = self.archive[np.argmin(d1 + 5 * d2, axis=0)]
        gbest = self.archive[self.rng.integers(0, len(self.archive), size=len(self.W))]
        return pbest, gbest

    def _operator(self, particle, pbest, gbest):
        rng = self.rng
        X, P, G, V = decs(particle), decs(pbest), decs(gbest), velocity(particle)
        N, D = X.shape
        W = rng.uniform(0.1, 0.5, (N, 1))
        r1, r2 = rng.random((N, 1)), rng.random((N, 1))
        C1, C2 = rng.uniform(1.5, 2.0, (N, 1)), rng.uniform(1.5, 2.0, (N, 1))
        pick = rng.random((N, 1)) < 0.7
        vel = W * V + np.where(pick, C1 * r1 * (P - X), C2 * r2 * (G - X))
        return self.evaluate(X + vel, V=vel)

    def step(self):
        pbest, gbest = self._get_best()
        self.swarm = self._operator(self.swarm, pbest, gbest)
        self.Z = np.minimum(self.Z, objs(self.swarm).min(axis=0))
        self.archive = _update_archive(Population.merge(self.archive, self.swarm), self.N)
        n = len(self.archive)
        partner = self.rng.integers(0, int(np.ceil(n / 2)), size=n)
        S = self.evaluate(ga_half(self.problem, np.vstack([decs(self.archive), decs(self.archive[partner])]), rng=self.rng))
        self.Z = np.minimum(self.Z, objs(S).min(axis=0))
        self.archive = _update_archive(Population.merge(self.archive, S), self.N)
        self.pop = self.archive
