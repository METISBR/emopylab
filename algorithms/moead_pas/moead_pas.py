# emopylab 2026
"""MOEA/D-PaS (Pareto adaptive scalarizing methods).

Reference:
R. Wang, Q. Zhang, and T. Zhang. IEEE TEC, 2016, 20(6): 821-837.
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

ALGORITHM_FLAGS = {"MOEADPaS": {"multi", "many", "real", "integer"}}


class MOEADPaS(LoopAlgorithm):
    """MOEA/D-DE with Pareto-adaptive scalarising: each subproblem uses an L_p scalarisation (p in 1..10 or inf)
    re-selected with probability 1-FE/maxFE as the one whose best solution lies closest to its weight vector."""

    PSET = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf])

    def __init__(self, pop_size=100, delta=0.9, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta = float(delta)

    def initial_size(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(n / 10))
        self.B = neighbors(self.W, self.T)
        return n

    def _znad(self, F):
        return F[nd_sort(F, None, 1)[0] == 1].max(axis=0)

    def start(self):
        F = objs(self.pop)
        self.pv = np.ones(self.pop_size)
        self.z = F.min(axis=0)
        self.znad = self._znad(F)

    def _g(self, F, P):
        with np.errstate(all="ignore"):
            Y = (F - self.z) / (self.znad - self.z) / self.W[P]
            pp = self.pv[P]
            g = np.zeros(len(P))
            inf = np.isinf(pp)
            g[inf] = Y[inf].max(axis=1)
            g[~inf] = np.sum(Y[~inf] ** pp[~inf, None], axis=1) ** (1.0 / pp[~inf])
        return g

    def _off_de(self, i, P):
        X = decs(self.pop)
        return self.evaluate(de(self.problem, X[[i]], X[[P[0]]], X[[P[1]]], rng=self.rng))

    def _off_ga(self, a, b):
        return self.evaluate(ga_half(self.problem, decs(self.pop[[int(a), int(b)]]), rng=self.rng))

    def _replace(self, idx, off):
        for j in idx:
            self.pop[int(j)] = off[0]

    def step(self):
        rng, N, M = self.rng, self.pop_size, self.problem.n_obj
        for i in range(N):
            P = self.B[i, rng.permutation(self.T)] if rng.random() < self.delta else rng.permutation(N)
            off = self._off_de(i, P)
            f = objs(off)[0]
            g_old = self._g(objs(self.pop[P]), P)
            g_new = self._g(np.tile(f, (len(P), 1)), P)
            self._replace(P[np.where(g_old > g_new)[0][: int(np.ceil(0.1 * self.T))]], off)
        F = objs(self.pop)
        self.z = np.minimum(self.z, F.min(axis=0))
        self.znad = self._znad(F)
        with np.errstate(all="ignore"):
            nObj = (F - self.z) / (self.znad - self.z)
            for i in np.where(rng.random(N) >= self.FE / self.max_FE)[0]:
                Y = nObj / self.W[i]
                g = np.column_stack([np.sum(Y ** q, axis=1) ** (1.0 / q) if np.isfinite(q) else Y.max(axis=1)
                                     for q in self.PSET])
                ZK = np.argmin(np.where(np.isnan(g), np.inf, g), axis=0)
                C = nObj[ZK]
                nc = np.sqrt(np.sum(C ** 2, axis=1))
                cos = C @ self.W[i] / (nc * np.linalg.norm(self.W[i]))
                d = np.sqrt(1 - cos ** 2) * nc
                self.pv[i] = self.PSET[0 if np.all(np.isnan(d)) else int(np.nanargmin(d))]
