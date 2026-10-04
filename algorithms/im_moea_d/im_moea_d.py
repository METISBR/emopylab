# emopylab 2026
"""IM-MOEA-D (inverse modeling MOEA/D).

Reference:
L. R. C. Farias and A. F. R. Araujo. IM-MOEA/D: An inverse modeling multi-objective evolutionary
algorithm based on decomposition. Proceedings of the IEEE International Conference on Systems, Mans
and Cybernetics, 2021.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, kmeans, neighbors_of, objs, uniform_point
from algorithms.community_utils.inverse_model import inverse_model_offspring
from core.population import Population

ALGORITHM_FLAGS = {'IMMOEAD': {'integer', 'large', 'multi', 'real'}}


class IMMOEAD(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, K: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.K = int(K)

    def initial_size(self):
        self.T = int(np.ceil(self.pop_size / 10))
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.Z = objs(self.pop).min(axis=0)

    def step(self):
        rng, N, T, W = self.rng, self.N, self.T, self.W
        part = kmeans(objs(self.pop), self.K, rng)
        blocks = [inverse_model_offspring(self, self.pop[part == k]) for k in np.unique(part)]
        off = self.evaluate(np.vstack(blocks))
        Fo = objs(off)
        self.Z = np.minimum(self.Z, Fo.min(axis=0))
        for i in range(len(off)):
            all_g = np.max(np.abs((Fo[i] - self.Z) * W), axis=1)
            chosen = int(np.where(all_g == all_g.min())[0][0])
            P = self.B[chosen][rng.permutation(T)]
            g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) * W[P], axis=1)
            g_new = np.max(np.abs(Fo[i] - self.Z) * W[P], axis=1)
            self.pop[P[np.where(g_old >= g_new)[0][:T]]] = off[i]
