# emopylab 2026
"""Constraint-MOEA/D (CMOEAD).

Reference:
H. Jain and K. Deb. IEEE TEC, 2014, 18(4): 602-622.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm

from algorithms.community_utils.moead_family import (
    Task,
    choose_parent_pool,
    cv_from_cons,
    gahalf_offspring,
    ind_F,
    ind_G,
    pbi_values,
    pop_F,
    population_cv,
    rng_from_algo,
    sample_initial,
    set_optimum_from_pop,
    weight_vectors,
    neighbors,
)


ALGORITHM_FLAGS = {"CMOEAD": {"multi", "many", "constrained", "real", "integer", "binary", "permutation", "label"}}


class CMOEAD(Algorithm):
    def __init__(self, pop_size=100, delta=0.9, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.delta = float(delta)
        self.sampling = sampling

    def _initialize_infill(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(self.pop_size / 10))
        self.nr = max(1, int(np.ceil(self.pop_size / 100)))
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
            off = gahalf_offspring(self.problem, self.pop, int(P[0]), int(P[1]), rng)
            self._tasks.append(Task(i=i, parents_pool=np.asarray(P, dtype=int)))
            offs.append(off)
        from algorithms.community_utils.moead_family import ensure_population
        return ensure_population(offs)

    def _advance(self, infills=None, **kwargs):
        for off, task in zip(infills, self._tasks):
            off_f = ind_F(off)
            off_g = ind_G(off)
            self.Z = np.minimum(self.Z, off_f)
            P = task.parents_pool
            cv_p = population_cv(self.pop[P])
            cv_o = cv_from_cons(off_g)
            g_old = pbi_values(pop_F(self.pop[P]), self.Z, self.W[P], theta=5.0)
            g_new = pbi_values(np.repeat(off_f[None, :], len(P), axis=0), self.Z, self.W[P], theta=5.0)
            cond = ((g_old >= g_new) & (cv_p == cv_o)) | (cv_p > cv_o)
            repl = np.where(cond)[0][: self.nr]
            if repl.size:
                self.pop[P[repl]] = off

    def _set_optimum(self):
        set_optimum_from_pop(self)

