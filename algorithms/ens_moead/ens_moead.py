# emopylab 2026
"""ENS-MOEA-D (ensemble neighborhood sizes).

Reference:
S. Zhao, P. N. Suganthan, and Q. Zhang. IEEE TEC, 2012, 16(3): 442-446.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm

from algorithms.community_utils.moead_family import (
    Task,
    RouletteWheelSelection,
    choose_dra_indices,
    choose_parent_pool,
    current_fe,
    de_offspring,
    ensure_population,
    ind_F,
    pop_F,
    rng_from_algo,
    sample_initial,
    set_optimum_from_pop,
    tchebycheff_values,
    update_pi_dra,
    weight_vectors,
    neighbors,
)


ALGORITHM_FLAGS = {"ENSMOEAD": {"multi", "many", "real", "integer"}}


class ENSMOEAD(Algorithm):
    def __init__(self, pop_size=100, NS=None, LP=50, delta=0.9, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.NS = np.asarray(NS if NS is not None else np.arange(25, 101, 25), dtype=int)
        self.LP = int(LP)
        self.delta = float(delta)
        self.sampling = sampling

    def _initialize_infill(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.nr = max(1, int(np.ceil(self.pop_size / 100)))
        self.B_full = neighbors(self.W, self.pop_size)
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.Z = np.min(pop_F(self.pop), axis=0)
        self.Pi = np.ones(self.pop_size, dtype=float)
        self.oldObj = tchebycheff_values(pop_F(self.pop), self.Z, self.W)
        self.p = np.ones(len(self.NS), dtype=float) / len(self.NS)
        self.FEs = np.zeros(len(self.NS), dtype=float)
        self.FEs_success = np.zeros(len(self.NS), dtype=float)

    def _infill(self):
        rng = rng_from_algo(self)
        self._tasks: list[Task] = []
        offs = []
        ns_sel = np.asarray(RouletteWheelSelection(self.pop_size, 1.0 / np.maximum(self.p, 1e-32), rng=rng), dtype=int) - 1
        for _ in range(5):
            I = choose_dra_indices(self.W, self.Pi, rng, self.pop_size)
            for i in I:
                k_local = int(self.NS[int(ns_sel[i])])
                P = choose_parent_pool(int(i), self.B_full, self.pop_size, rng, self.delta, local_size=k_local)
                off = de_offspring(self.problem, self.pop, int(i), P, rng)
                self._tasks.append(Task(i=int(i), parents_pool=np.asarray(P, dtype=int), ns_idx=int(ns_sel[i])))
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
                self.FEs_success[int(task.ns_idx)] += 1
            self.FEs[int(task.ns_idx)] += 1
        n_gen_proxy = int(np.ceil(max(current_fe(self), 1) / max(self.pop_size, 1)))
        if n_gen_proxy % 10 == 0:
            self.Pi, self.oldObj = update_pi_dra(pop_F(self.pop), self.W, self.Z, self.Pi, self.oldObj)
        if self.LP > 0 and n_gen_proxy % self.LP == 0:
            R = self.FEs_success / np.maximum(self.FEs, 1.0)
            if np.sum(R) > 0:
                self.p = R / np.sum(R)
            self.FEs[:] = 0.0
            self.FEs_success[:] = 0.0

    def _set_optimum(self):
        set_optimum_from_pop(self)

