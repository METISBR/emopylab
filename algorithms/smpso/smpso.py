# emopylab 2026
"""SMPSO (speed-constrained multi-objective particle swarm optimization).

Reference:
A. J. Nebro, J. J. Durillo, J. Garcia-Nieto, C. A. Coello Coello, F. Luna, and E. Alba. SMPSO: A new
PSO-based metaheuristic for multi-objective optimization. Proceedings of the IEEE Symposium on
Computational Intelligence in Multi-Criteria Decision-Making, 2009, 66-73.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, first_front, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'SMPSO': {'integer', 'multi', 'real'}}


def _velocity(pop):
    v = pop.get("V")
    try:
        v = np.asarray(v, dtype=float)
        if v.ndim == 2 and v.shape[0] == len(pop):
            return v
    except (TypeError, ValueError):
        pass
    return np.zeros((len(pop), np.asarray(pop.get("X")).shape[1]))


def _update_gbest(gbest, N):
    gbest = gbest[first_front(objs(gbest))]
    cd = crowding(objs(gbest))
    rank = np.argsort(-cd, kind="stable")[: min(N, len(gbest))]
    return gbest[rank], cd[rank]


def _update_pbest(pbest, pop):
    replace = ~np.all(objs(pop) >= objs(pbest), axis=1)
    out = pbest.copy(deep=False)
    out[replace] = pop[replace]
    return out


class SMPSO(LoopAlgorithm):
    def _operator(self, particle, pbest, gbest):
        rng = self.rng
        X, P, G = decs(particle), decs(pbest), decs(gbest)
        V = _velocity(particle)
        N, D = X.shape
        lower, upper = self.lower, self.upper
        W = rng.uniform(0.1, 0.5, (N, 1))
        r1, r2 = rng.random((N, 1)), rng.random((N, 1))
        C1, C2 = rng.uniform(1.5, 2.5, (N, 1)), rng.uniform(1.5, 2.5, (N, 1))
        vel = W * V + C1 * r1 * (P - X) + C2 * r2 * (G - X)
        phi = np.maximum(4.0, C1 + C2)
        vel = vel * 2.0 / np.abs(2.0 - phi - np.sqrt(phi ** 2 - 4.0 * phi))
        delta = (upper - lower) / 2.0
        vel = np.maximum(np.minimum(vel, delta), -delta)
        dec = X + vel
        repair = (dec < lower) | (dec > upper)
        vel[repair] = 0.001 * vel[repair]
        dec = np.maximum(np.minimum(dec, upper), lower)
        disM = 20.0
        site1 = np.repeat(rng.random((N, 1)) < 0.15, D, axis=1)
        site2 = rng.random((N, D)) < 1.0 / D
        mu = rng.random((N, D))
        span = np.broadcast_to(upper - lower, (N, D))
        lo = np.broadcast_to(lower, (N, D))
        up = np.broadcast_to(upper, (N, D))
        with np.errstate(all="ignore"):
            t = site1 & site2 & (mu <= 0.5)
            dec[t] = dec[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (dec[t] - lo[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
            t = site1 & site2 & (mu > 0.5)
            dec[t] = dec[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - dec[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
        return self.evaluate(dec, V=vel)

    def start(self):
        self.particles = self.pop
        self.pbest = self.pop
        self.gbest, self.crowd = _update_gbest(self.pop, self.N)
        self.pop = self.gbest

    def step(self):
        leaders = self.gbest[tournament(2, self.N, -self.crowd, rng=self.rng)]
        self.particles = self._operator(self.particles, self.pbest, leaders)
        self.gbest, self.crowd = _update_gbest(Population.merge(self.gbest, self.particles), self.N)
        self.pbest = _update_pbest(self.pbest, self.particles)
        self.pop = self.gbest
