# emopylab 2026
"""MOEA-D-DE (MOEA/D based on differential evolution).

Reference:
H. Li and Q. Zhang. IEEE TEC, 2009, 13(2): 284-302.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, objs
from algorithms.community_utils.moead_family import neighbors, tchebycheff_values, weight_vectors

ALGORITHM_FLAGS = {"MOEADDE": {"multi", "many", "real", "integer"}}


class MOEADDE(LoopAlgorithm):
    """Steady-state MOEA/D: each subproblem creates one DE offspring from its neighbourhood (probability ``delta``)
    or the whole population, which then replaces at most ``nr`` solutions it improves in Tchebycheff value."""

    def __init__(self, pop_size=100, delta=0.9, nr=2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta = float(delta)
        self.nr = int(nr)

    def initial_size(self) -> int:
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(n / 10))
        self.B = neighbors(self.W, self.T)
        return n

    def start(self):
        self.Z = objs(self.pop).min(axis=0)

    def step(self):
        rng, N = self.rng, self.pop_size
        for i in range(N):
            P = self.B[i, rng.permutation(self.T)] if rng.random() < self.delta else rng.permutation(N)
            X = decs(self.pop)
            off = self.evaluate(de(self.problem, X[[i]], X[[P[0]]], X[[P[1]]], rng=rng))
            f = objs(off)[0]
            self.Z = np.minimum(self.Z, f)
            g_old = tchebycheff_values(objs(self.pop[P]), self.Z, self.W[P])
            g_new = tchebycheff_values(np.tile(f, (len(P), 1)), self.Z, self.W[P])
            for j in P[np.where(g_old >= g_new)[0][: self.nr]]:
                self.pop[int(j)] = off[0]
