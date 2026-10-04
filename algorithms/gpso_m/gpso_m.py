# emopylab 2026
"""GPSO-M (gradient based particle swarm optimization algorithm (for multi-objective optimization)).

Reference:
M. M. Noel. A new gradient based particle swarm optimization algorithm for accurate computation of
global minimum. Applied Soft Computing, 2012, 12: 353-359.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, adds, cons, decs, first_front, gradient_direction, objs, uniform_point, velocity
from core.population import Population

ALGORITHM_FLAGS = {'GPSOM': {'constrained', 'integer', 'many', 'multi', 'real'}}


class GPSOM(LoopAlgorithm):
    """One PSO sub-swarm per weight vector whose global best is refined by a gradient local search."""

    def __init__(self, pop_size: int = 100, popsize: int = 20, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.popsize = int(popsize)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.popsize

    def _local_search(self, pos, w, max_iter=5, tol=1e-3):
        step, k, error = 1.0, 1, 10.0
        lo, up = self.lower, self.upper
        while error > tol and k < max_iter:
            g1, _ = gradient_direction(self, pos, w)
            dec = np.minimum(np.maximum(np.asarray(pos.X, dtype=float) - step * g1, lo), up)
            off = self.evaluate(dec[None, :])[0]
            g2, _ = gradient_direction(self, off, w)
            dg = g2 - g1
            with np.errstate(all="ignore"):
                step = abs((np.asarray(off.X, float) - np.asarray(pos.X, float)) @ dg) / np.linalg.norm(dg) ** 2
            error = np.linalg.norm(np.asarray(off.X, float) - np.asarray(pos.X, float))
            pos, k = off, k + 1
        return pos

    def start(self):
        N, W = self.N, self.W
        self.subpops = [self.pop]
        self.pbests = [self.pop]
        self.gbest = []
        for i in range(N):
            if i > 0:
                sp = self.evaluate(self.random_decs(self.popsize))
                self.subpops.append(sp)
                self.pbests.append(sp)
            front = first_front(objs(self.subpops[i]), cons(self.subpops[i]) if cons(self.subpops[i]).size else None)
            pool = self.subpops[0][front] if i > 0 else self.subpops[0][front]      # (sic) the reference reads sub-swarm 1
            self.gbest.append(self._local_search(pool[0], W[i]))
        self.pop = Population.create(self.gbest)

    def _update_pbest(self, pb, sw):
        temp = objs(pb) - objs(sw)
        dom = np.any(temp < 0, axis=1).astype(int) - np.any(temp > 0, axis=1).astype(int)
        rep = (dom == -1) | ((dom == 0) & (self.rng.random(len(dom)) < 0.5))
        out = pb.copy(deep=False)
        out[rep] = sw[rep]
        return out

    def step(self):
        rng = self.rng
        for i in range(self.N):
            sp, pb = self.subpops[i].copy(deep=False), self.pbests[i]
            for j in range(self.popsize):
                X, P, G = decs(sp[j:j + 1]), decs(pb[j:j + 1]), decs(Population.create([self.gbest[i]]))
                V = velocity(sp[j:j + 1])
                vel = 0.4 * V + rng.random(X.shape) * (P - X) + rng.random(X.shape) * (G - X)
                sp[j] = self.evaluate(X + vel, V=vel)[0]
                pb[j] = self._update_pbest(pb[j:j + 1], sp[j:j + 1])[0]
            self.subpops[i], self.pbests[i] = sp, pb
            front = first_front(objs(sp), cons(sp) if cons(sp).size else None)
            self.gbest[i] = self._local_search(sp[front][0], self.W[i])
        self.pop = Population.create(self.gbest)
