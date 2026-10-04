# emopylab 2026
"""Constraint-MOEA/D (CMOEAD).

Reference:
H. Jain and K. Deb. IEEE TEC, 2014, 18(4): 602-622.
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

ALGORITHM_FLAGS = {"CMOEAD": {"multi", "many", "constrained", "real", "integer", "binary", "permutation", "label"}}


class CMOEAD(LoopAlgorithm):
    """Steady-state MOEA/D with PBI (theta 5): an offspring replaces at most ``nr`` neighbours that violate the
    constraints more, or equally with a worse PBI value."""

    def __init__(self, pop_size=100, delta=0.9, sampling=None, ref_dirs=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta = float(delta)
        self.ref_dirs = ref_dirs

    def initial_size(self):
        if self.ref_dirs is not None:
            self.W = np.asarray(self.ref_dirs, dtype=float)
        else:
            self.W = weight_vectors(self.pop_size, self.problem.n_obj)[0]
        self.pop_size = n = len(self.W)
        self.T = int(np.ceil(n / 10))
        self.nr = int(np.ceil(n / 100))
        self.B = neighbors(self.W, self.T)
        return n

    def start(self):
        self.Z = objs(self.pop).min(axis=0)

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
        for i in range(N):
            P = self.B[i, rng.permutation(self.T)] if rng.random() < self.delta else rng.permutation(N)
            off = self._off_ga(P[0], P[1])
            f = objs(off)[0]
            self.Z = np.minimum(self.Z, f)
            cvo, cvp = cv(off)[0], cv(self.pop[P])
            g_old = pbi_values(objs(self.pop[P]), self.Z, self.W[P], 5.0)
            g_new = pbi_values(np.tile(f, (len(P), 1)), self.Z, self.W[P], 5.0)
            self._replace(P[np.where(((g_old >= g_new) & (cvp == cvo)) | (cvp > cvo))[0][: self.nr]], off)
