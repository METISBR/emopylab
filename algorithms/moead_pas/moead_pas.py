# emopylab 2026
"""MOEA/D-PaS (Pareto adaptive scalarizing methods).

Reference:
R. Wang, Q. Zhang, and T. Zhang. IEEE TEC, 2016, 20(6): 821-837.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm

from operators.utility_functions.NDSort import NDSort
from algorithms.community_utils.moead_family import (
    Task,
    choose_parent_pool,
    de_offspring,
    ensure_population,
    fe_ratio,
    ind_F,
    neighbors,
    pop_F,
    rng_from_algo,
    sample_initial,
    set_optimum_from_pop,
    weight_vectors,
)


ALGORITHM_FLAGS = {"MOEADPaS": {"multi", "many", "real", "integer"}}


class MOEADPaS(Algorithm):
    def __init__(self, pop_size=100, delta=0.9, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.delta = float(delta)
        self.sampling = sampling

    def _initialize_infill(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors(self.W, self.T)
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.p = np.ones(self.pop_size, dtype=float)
        F = pop_F(self.pop)
        self.z = np.min(F, axis=0)
        front = np.asarray(NDSort(F, 1)[0], dtype=float)
        nd = np.where(front == 1)[0]
        self.znad = np.max(F[nd], axis=0) if nd.size else np.max(F, axis=0)

    def _infill(self):
        rng = rng_from_algo(self)
        self._tasks: list[Task] = []
        offs = []
        for i in range(self.pop_size):
            P = choose_parent_pool(i, self.B, self.pop_size, rng, self.delta)
            off = de_offspring(self.problem, self.pop, i, P, rng)
            self._tasks.append(Task(i=i, parents_pool=np.asarray(P, dtype=int)))
            offs.append(off)
        return ensure_population(offs)

    def _pas_vals(self, F: np.ndarray, idx: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        if F.ndim == 1:
            F = F[None, :]
        den = np.maximum(self.znad - self.z, 1e-32)
        Y = np.abs((F - self.z) / den) / np.maximum(self.W[idx], 1e-12)
        pvals = self.p[idx]
        out = np.zeros(Y.shape[0], dtype=float)
        inf_mask = np.isinf(pvals)
        if np.any(inf_mask):
            out[inf_mask] = np.max(Y[inf_mask], axis=1)
        if np.any(~inf_mask):
            pe = pvals[~inf_mask]
            out[~inf_mask] = np.sum(Y[~inf_mask] ** pe[:, None], axis=1) ** (1.0 / pe)
        return out

    def _advance(self, infills=None, **kwargs):
        for off, task in zip(infills, self._tasks):
            off_f = ind_F(off)
            P = task.parents_pool
            g_old = self._pas_vals(pop_F(self.pop[P]), P)
            g_new = self._pas_vals(np.repeat(off_f[None, :], len(P), axis=0), P)
            repl = np.where(g_old > g_new)[0][: max(1, int(np.ceil(0.1 * len(P))))]
            if repl.size:
                self.pop[P[repl]] = off

        F = pop_F(self.pop)
        self.z = np.minimum(self.z, np.min(F, axis=0))
        front = np.asarray(NDSort(F, 1)[0], dtype=float)
        nd = np.where(front == 1)[0]
        self.znad = np.max(F[nd], axis=0) if nd.size else np.max(F, axis=0)

        Pset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf], dtype=float)
        nObj = (F - self.z) / np.maximum(self.znad - self.z, 1e-32)
        ratio = fe_ratio(self)
        rng = rng_from_algo(self)
        for i in np.where(rng.random(self.pop_size) >= ratio)[0]:
            Y = nObj / np.maximum(self.W[i], 1e-12)
            g = np.zeros((self.pop_size, len(Pset)), dtype=float)
            for j, pv in enumerate(Pset):
                if np.isfinite(pv):
                    g[:, j] = np.sum(Y ** pv, axis=1) ** (1.0 / pv)
                else:
                    g[:, j] = np.max(Y, axis=1)
            ZK = np.argmin(g, axis=0)
            cand = nObj[ZK]
            norms = np.maximum(np.linalg.norm(cand, axis=1), 1e-32)
            wnorm = max(np.linalg.norm(self.W[i]), 1e-32)
            cos = np.sum(cand * self.W[i], axis=1) / (norms * wnorm)
            zidx = int(np.argmin(np.sqrt(np.maximum(0.0, 1.0 - cos * cos)) * norms))
            self.p[i] = Pset[zidx]

    def _set_optimum(self):
        set_optimum_from_pop(self)

