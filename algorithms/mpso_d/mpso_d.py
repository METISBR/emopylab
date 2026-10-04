# emopylab 2026
"""MPSO-D (multi-objective particle swarm optimization algorithm based on).

Reference:
C. Dai, Y. Wang, and M. Ye. A new multi-objective particle swarm optimization algorithm based on
decomposition. Information Sciences, 2015, 325: 541-557.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, neighbors_of, objs, tournament, uniform_point, velocity
from algorithms.mmopso.mmopso import MMOPSO
from core.population import Population

ALGORITHM_FLAGS = {'MPSOD': {'integer', 'many', 'multi', 'real'}}


class MPSOD(LoopAlgorithm):
    _classification = MMOPSO._classification

    def initial_size(self):
        W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return 2 * self.pop_size

    def start(self):
        self.Z = objs(self.pop).min(axis=0)
        self.pop = self._classification(self.pop)

    def _mating_selection(self):
        rng, N, W, B = self.rng, self.N, self.W, self.B
        F = objs(self.pop) - self.Z
        parent = tournament(2, N, -crowding(F), rng=rng)
        pbest, gbest = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
        for i in range(N):
            P = B[i] if rng.random() < 0.9 else np.arange(N)
            pbest[i] = P[rng.integers(0, len(P))]
            score = (F[P] @ W[P].mean(axis=0)) / np.sum(F[P] ** 2, axis=1) ** 0.6
            gbest[i] = P[int(np.argmax(score))]
        return parent, pbest, gbest

    def _operator(self, particle, pbest, gbest, c1=2.0, c2=2.0, CR=0.5, F=0.5, proM=1.0, disM=20.0):
        rng = self.rng
        X, P, G, V = decs(particle), decs(pbest), decs(gbest), velocity(particle)
        N, D = X.shape
        lo, up = self.lower, self.upper
        do_pso = np.repeat(rng.random((N, 1)) < 0.5, D, axis=1)
        w = 0.9 - self.FE / self.max_FE * 0.8
        r1, r2 = rng.random((N, 1)), rng.random((N, 1))
        vel, dec = V.copy(), X.copy()
        pv = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
        vel[do_pso] = pv[do_pso]
        dec[do_pso] = (X + vel)[do_pso]
        invalid = (dec < lo) | (dec > up)
        dec[invalid] = X[invalid]
        site = ~do_pso & (rng.random((N, D)) < CR)
        dec[site] = (X + F * (G - P))[site]
        dec = np.maximum(np.minimum(dec, up), lo)
        s2 = rng.random((N, D)) < proM / D
        mu = rng.random((N, D))
        span = np.broadcast_to(up - lo, (N, D))
        lo_b, up_b = np.broadcast_to(lo, (N, D)), np.broadcast_to(up, (N, D))
        t = s2 & (mu <= 0.5)
        dec[t] = dec[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (dec[t] - lo_b[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = s2 & (mu > 0.5)
        dec[t] = dec[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up_b[t] - dec[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
        return self.evaluate(dec, V=vel)

    def step(self):
        parent, pbest, gbest = self._mating_selection()
        off = self._operator(self.pop[parent], self.pop[pbest], self.pop[gbest])
        self.Z = np.minimum(self.Z, objs(off).min(axis=0))
        self.pop = self._classification(Population.merge(self.pop, off))
