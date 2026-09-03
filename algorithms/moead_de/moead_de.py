# emopylab 2026
"""MOEA-D-DE (MOEA/D based on differential evolution).

Reference:
H. Li and Q. Zhang. IEEE TEC, 2009, 13(2): 284-302.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm

from algorithms.community_utils.moead_family import (
    Task,
    choose_parent_pool,
    de_offspring,
    ensure_population,
    ind_F,
    pop_F,
    rng_from_algo,
    sample_initial,
    set_optimum_from_pop,
    tchebycheff_values,
    weight_vectors,
    neighbors,
)


ALGORITHM_FLAGS = {"MOEADDE": {"multi", "many", "real", "integer"}}


class MOEADDE(Algorithm):
    def __init__(self, pop_size=100, delta=0.9, nr=2, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.delta = float(delta)
        self.nr = int(nr)
        self.sampling = sampling

    def _initialize_infill(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors(self.W, self.T)
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.Z = np.min(pop_F(self.pop), axis=0)

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

    def _advance(self, infills=None, **kwargs):
        for off, task in zip(infills, self._tasks):
            off_f = ind_F(off)
            self.Z = np.minimum(self.Z, off_f)
            P = task.parents_pool
            g_old = tchebycheff_values(pop_F(self.pop[P]), self.Z, self.W[P])
            g_new = tchebycheff_values(np.repeat(off_f[None, :], len(P), axis=0), self.Z, self.W[P])
            repl = np.where(g_old >= g_new)[0][: self.nr]
            if repl.size:
                self.pop[P[repl]] = off

    def _set_optimum(self):
        set_optimum_from_pop(self)

