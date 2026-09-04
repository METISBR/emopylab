# emopylab 2026
"""MOEA/D-DCWV (distribution control of weight vector set).

Reference:
T. Takagi et al., BICT 2019.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm

from algorithms.community_utils.moead_family import (
    Task,
    ensure_population,
    gahalf_offspring,
    ind_F,
    neighbors,
    normalized_tchebycheff_values,
    pop_F,
    rng_from_algo,
    sample_initial,
    set_optimum_from_pop,
    set_weight_dcwv,
    update_weight_dcwv,
    weight_vectors,
)


ALGORITHM_FLAGS = {"MOEADDCWV": {"multi", "many", "real", "integer", "label", "binary", "permutation"}}


class MOEADDCWV(Algorithm):
    def __init__(self, pop_size=100, p=-1.0, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.p = float(p)
        self.sampling = sampling

    def _initialize_infill(self):
        W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(self.pop_size / 10))
        self.W, self.W0 = set_weight_dcwv(W, self.p)
        self.B = neighbors(self.W, self.T)
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.Z = np.min(pop_F(self.pop), axis=0)

    def _infill(self):
        rng = rng_from_algo(self)
        if self.W0.size:
            self.W = update_weight_dcwv(pop_F(self.pop), self.W0)
            self.B = neighbors(self.W, self.T)
        self._tasks: list[Task] = []
        offs = []
        for i in range(self.pop_size):
            P = self.B[i, rng.permutation(self.B.shape[1])]
            off = gahalf_offspring(self.problem, self.pop, int(P[0]), int(P[1]), rng)
            self._tasks.append(Task(i=i, parents_pool=np.asarray(P, dtype=int)))
            offs.append(off)
        return ensure_population(offs)

    def _advance(self, infills=None, **kwargs):
        for off, task in zip(infills, self._tasks):
            off_f = ind_F(off)
            self.Z = np.minimum(self.Z, off_f)
            Zmax = np.max(pop_F(self.pop), axis=0)
            P = task.parents_pool
            g_old = normalized_tchebycheff_values(pop_F(self.pop[P]), self.Z, Zmax, self.W[P])
            g_new = normalized_tchebycheff_values(np.repeat(off_f[None, :], len(P), axis=0), self.Z, Zmax, self.W[P])
            repl = np.where(g_old >= g_new)[0]
            if repl.size:
                self.pop[P[repl]] = off

    def _set_optimum(self):
        set_optimum_from_pop(self)

