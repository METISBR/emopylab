# emopylab 2026
"""MOEA/D-DCWV (distribution control of weight vector set).

Reference:
T. Takagi et al., BICT 2019.
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

ALGORITHM_FLAGS = {"MOEADDCWV": {"multi", "many", "real", "integer", "label", "binary", "permutation"}}


class MOEADDCWV(LoopAlgorithm):
    """MOEA/D with a distribution-controlled weight vector set (fixed by ``p`` in [0,1], otherwise re-estimated
    every generation from the population); normalised Tchebycheff replacement of every improved neighbour.  The
    neighbourhoods stay those of the initial weights."""

    def __init__(self, pop_size=100, p=-1.0, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.p = float(p)

    def initial_size(self):
        W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(n / 10))
        self.W, self.W0 = set_weight_dcwv(W, self.p)
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
        if self.W0.size:
            self.W = update_weight_dcwv(objs(self.pop), self.W0)
        for i in range(N):
            P = self.B[i, rng.permutation(self.T)]
            off = self._off_ga(P[0], P[1])
            f = objs(off)[0]
            self.Z = np.minimum(self.Z, f)
            zmax = objs(self.pop).max(axis=0)
            with np.errstate(all="ignore"):
                g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) / (zmax - self.Z) / self.W[P], axis=1)
                g_new = np.max(np.abs(f - self.Z) / (zmax - self.Z) / self.W[P], axis=1)
            self._replace(P[g_old >= g_new], off)
