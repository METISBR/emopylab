# emopylab 2026
"""dMOPSO (mOPSO based on decomposition).

Reference:
S. Z. Martinez and C. A. Coello Coello. A multi-objective particle swarm optimizer based on
decomposition. Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2011,
69-76.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, objs, uniform_point, velocity
from core.population import Population

ALGORITHM_FLAGS = {'dMOPSO': {'integer', 'multi', 'real'}}


def _pbi(F, Z, W):
    G = F - Z
    cos = 1.0 - cosine_distance(G, W)
    norm = np.linalg.norm(G, axis=1)
    return norm[:, None] * (cos + 5 * np.sqrt(np.maximum(0.0, 1 - cos ** 2)))


def _update_gbest(W, pop, Z):
    pbi = _pbi(objs(pop), Z, W)
    remain = list(range(len(pop)))
    assoc = np.zeros(len(W), dtype=int)
    for i in range(len(W)):
        b = int(np.argmin(pbi[remain, i]))
        assoc[i] = remain.pop(b)
    return pop[assoc]


class dMOPSO(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, Ta: int = 2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Ta = int(Ta)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        self.swarm = self.pop
        self.age = np.zeros(self.N)
        self.Z = objs(self.pop).min(axis=0)
        self.pbest = self.pop.copy(deep=False)
        self.gbest = _update_gbest(self.W, self.pop, self.Z)
        self.pop = self.gbest

    def _operator(self, particle, pbest, gbest, w=0.4):
        rng = self.rng
        X, P, G, V = decs(particle), decs(pbest), decs(gbest), velocity(particle)
        N = len(X)
        vel = w * V + rng.random((N, 1)) * (P - X) + rng.random((N, 1)) * (G - X)
        dec = X + vel
        repair = (dec < self.lower) | (dec > self.upper)
        vel[repair] = -vel[repair]
        return self.evaluate(dec, V=vel)

    def step(self):
        N, W = self.N, self.W
        act = np.where(self.age < self.Ta)[0]
        if len(act):
            self.swarm[act] = self._operator(self.swarm[act], self.pbest[act], self.gbest[act])
        for i in np.where(self.age >= self.Ta)[0]:
            g, p = decs(self.gbest[i:i + 1])[0], decs(self.pbest[i:i + 1])[0]
            self.swarm[i] = self.evaluate(((g - p) / 2 + self.rng.standard_normal(self.D) * np.abs(g - p))[None, :])[0]
            self.age[i] = 0
        self.Z = np.minimum(self.Z, objs(self.swarm).min(axis=0))
        pb, sw = objs(self.pbest) - self.Z, objs(self.swarm) - self.Z
        normW = np.linalg.norm(W, axis=1)
        npb, nsw = np.linalg.norm(pb, axis=1), np.linalg.norm(sw, axis=1)
        with np.errstate(all="ignore"):
            cpb = np.sum(pb * W, axis=1) / normW / npb
            csw = np.sum(sw * W, axis=1) / normW / nsw
            g_old = npb * cpb + 5 * npb * np.sqrt(1 - cpb ** 2)
            g_new = nsw * csw + 5 * nsw * np.sqrt(1 - csw ** 2)
        better = g_new <= g_old
        self.pbest[better] = self.swarm[better]
        self.age[better] = 0
        self.age[~better] += 1
        self.gbest = _update_gbest(W, Population.merge(self.gbest, self.swarm), self.Z)
        self.pop = self.gbest
