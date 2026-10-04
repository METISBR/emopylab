# emopylab 2026
"""MOEA/D-DU (distance based updating strategy).

Reference:
Y. Yuan et al. IEEE TEC, 2016, 20(2): 180-198.
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

ALGORITHM_FLAGS = {"MOEADDU": {"multi", "many", "real", "integer", "label", "binary", "permutation"}}


class MOEADDU(LoopAlgorithm):
    """MOEA/D with a distance-based updating strategy: each offspring may replace (at most once) one of the ``K``
    subproblems whose weight vectors are closest in angle to it, judged by the normalised Tchebycheff value."""

    def __init__(self, pop_size=100, delta=0.9, K=5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta = float(delta)
        self.K = int(K)

    def initial_size(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(n / 10))
        self.B = neighbors(self.W, self.T)
        return n

    def start(self):
        F = objs(self.pop)
        self.z, self.znad = F.min(axis=0), F.max(axis=0)

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
        _, self.z, self.znad = normalize_du(objs(self.pop), self.z, self.znad)
        for i in range(N):
            j = self.B[i, rng.integers(self.T)] if rng.random() < self.delta else rng.integers(N)
            off = self._off_ga(i, j)
            f = objs(off)[0]
            with np.errstate(all="ignore"):
                cos = self.W @ f / (np.linalg.norm(self.W, axis=1) * np.linalg.norm(f))
            P = np.argsort(-cos, kind="stable")[: self.K]
            with np.errstate(all="ignore"):
                g_old = np.max(np.abs(objs(self.pop[P]) - self.z) / (self.znad - self.z) / self.W[P], axis=1)
                g_new = np.max(np.abs(f - self.z) / (self.znad - self.z) / self.W[P], axis=1)
            self._replace(P[np.where(g_old >= g_new)[0][:1]], off)
