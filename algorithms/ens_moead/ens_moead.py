# emopylab 2026
"""ENS-MOEA-D (ensemble neighborhood sizes).

Reference:
S. Zhao, P. N. Suganthan, and Q. Zhang. IEEE TEC, 2012, 16(3): 442-446.
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

ALGORITHM_FLAGS = {"ENSMOEAD": {"multi", "many", "real", "integer"}}


class ENSMOEAD(LoopAlgorithm):
    """MOEA/D-DRA with an ensemble of neighbourhood sizes ``NS`` chosen per subproblem by roulette on their recent
    success rates (reset every ``LP`` generations).  Neighbourhoods larger than the population use all of it."""

    def __init__(self, pop_size=100, NS=None, LP=50, delta=0.9, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.NS = np.asarray(NS if NS is not None else np.arange(25, 101, 25), dtype=int)
        self.LP = int(LP)
        self.delta = float(delta)

    def initial_size(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.nr = int(np.ceil(n / 100))
        self.B = neighbors(self.W, n)
        return n

    def start(self):
        F = objs(self.pop)
        self.Z = F.min(axis=0)
        self.Pi = np.ones(self.pop_size)
        self.oldObj = tchebycheff_values(F, self.Z, self.W)
        self.p = np.ones(len(self.NS)) / len(self.NS)
        self.FEs = np.zeros(len(self.NS))
        self.FEs_success = np.zeros(len(self.NS))

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
        ns = roulette(N, 1.0 / self.p, rng=rng)
        for _ in range(5):
            for i in choose_dra_indices(self.W, self.Pi, rng, N):
                k = int(ns[i])
                if rng.random() < self.delta:
                    P = self.B[i, rng.permutation(min(int(self.NS[k]), N))]
                else:
                    P = rng.permutation(N)
                off = self._off_de(int(i), P)
                f = objs(off)[0]
                self.Z = np.minimum(self.Z, f)
                g_old = tchebycheff_values(objs(self.pop[P]), self.Z, self.W[P])
                g_new = tchebycheff_values(np.tile(f, (len(P), 1)), self.Z, self.W[P])
                rep = np.where(g_old >= g_new)[0][: self.nr]
                self._replace(P[rep], off)
                if len(rep):
                    self.FEs_success[k] += 1
                self.FEs[k] += 1
        g = int(np.ceil(self.FE / N))
        if g % 10 == 0:
            self.Pi, self.oldObj = update_pi_dra(objs(self.pop), self.W, self.Z, self.Pi, self.oldObj)
        if g % self.LP == 0:
            with np.errstate(all="ignore"):
                R = self.FEs_success / self.FEs
            if np.all(np.isfinite(R)) and R.sum() > 0:      # an undefined rate keeps the previous probabilities
                self.p = R / R.sum()
            self.FEs[:] = 0
            self.FEs_success[:] = 0
