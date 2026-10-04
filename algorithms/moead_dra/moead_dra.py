# emopylab 2026
"""MOEA/D-DRA (dynamical resource allocation).

Reference:
Q. Zhang, W. Liu, and H. Li. CEC 2009.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cv, de, decs, ga_half, nd_sort, objs, roulette
from algorithms.community_utils.moead_family import (
    choose_dra_indices,
    neighbors,
    normalize_du,
    pbi_values,
    set_weight_dcwv,
    tchebycheff_values,
    update_pi_dra,
    update_weight_dcwv,
    weight_vectors,
)

ALGORITHM_FLAGS = {"MOEADDRA": {"multi", "many", "real", "integer"}}


class MOEADDRA(LoopAlgorithm):
    """MOEA/D-DE with dynamical resource allocation: five sub-generations per generation, each over the boundary
    subproblems plus N/5 chosen by 10-tournament on the utility ``Pi`` (updated every 10 generations)."""

    def __init__(self, pop_size=100, delta=0.9, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta = float(delta)

    def initial_size(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(n / 10))
        self.nr = int(np.ceil(n / 100))
        self.B = neighbors(self.W, self.T)
        return n

    def start(self):
        F = objs(self.pop)
        self.Z = F.min(axis=0)
        self.Pi = np.ones(self.pop_size)
        self.oldObj = tchebycheff_values(F, self.Z, self.W)

    def _off_de(self, i, P):
        X = decs(self.pop)
        return self.evaluate(de(self.problem, X[[i]], X[[P[0]]], X[[P[1]]], rng=self.rng))

    def _off_ga(self, a, b):
        return self.evaluate(ga_half(self.problem, decs(self.pop[[int(a), int(b)]]), rng=self.rng))

    def _replace(self, idx, off):
        for j in idx:
            self.pop[int(j)] = off[0]

    def step(self):
        rng, N = self.rng, self.pop_size
        for _ in range(5):
            for i in choose_dra_indices(self.W, self.Pi, rng, N):
                P = self.B[i, rng.permutation(self.T)] if rng.random() < self.delta else rng.permutation(N)
                off = self._off_de(int(i), P)
                f = objs(off)[0]
                self.Z = np.minimum(self.Z, f)
                g_old = tchebycheff_values(objs(self.pop[P]), self.Z, self.W[P])
                g_new = tchebycheff_values(np.tile(f, (len(P), 1)), self.Z, self.W[P])
                self._replace(P[np.where(g_old >= g_new)[0][: self.nr]], off)
        if int(np.ceil(self.FE / N)) % 10 == 0:
            self.Pi, self.oldObj = update_pi_dra(objs(self.pop), self.W, self.Z, self.Pi, self.oldObj)
