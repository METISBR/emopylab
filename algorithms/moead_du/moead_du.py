# emopylab 2026
"""MOEA/D-DU (distance based updating strategy).

Reference:
Y. Yuan et al. IEEE TEC, 2016, 20(2): 180-198.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm

from algorithms.community_utils.moead_family import (
    Task,
    choose_parent_pool,
    ensure_population,
    gahalf_offspring,
    ind_F,
    neighbors,
    normalize_du,
    normalized_tchebycheff_values,
    pop_F,
    rng_from_algo,
    sample_initial,
    set_optimum_from_pop,
    weight_vectors,
)


ALGORITHM_FLAGS = {"MOEADDU": {"multi", "many", "real", "integer", "label", "binary", "permutation"}}


class MOEADDU(Algorithm):
    def __init__(self, pop_size=100, delta=0.9, K=5, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.delta = float(delta)
        self.K = int(K)
        self.sampling = sampling

    def _initialize_infill(self):
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = n
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors(self.W, self.T)
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        F = pop_F(self.pop)
        self.z = np.min(F, axis=0)
        self.znad = np.max(F, axis=0)

    def _infill(self):
        rng = rng_from_algo(self)
        _, self.z, self.znad = normalize_du(pop_F(self.pop), self.z, self.znad)
        self._tasks: list[Task] = []
        offs = []
        for i in range(self.pop_size):
            if rng.random() < self.delta:
                P = np.array([int(self.B[i, rng.integers(0, self.B.shape[1])])], dtype=int)
            else:
                P = np.array([int(rng.integers(0, self.pop_size))], dtype=int)
            off = gahalf_offspring(self.problem, self.pop, i, int(P[0]), rng)
            self._tasks.append(Task(i=i, parents_pool=P))
            offs.append(off)
        return ensure_population(offs)

    def _advance(self, infills=None, **kwargs):
        for off in infills:
            off_f = ind_F(off)
            num = self.W @ off_f
            den = np.maximum(np.linalg.norm(self.W, axis=1) * np.linalg.norm(off_f), 1e-32)
            rank = np.argsort(-(num / den))
            P = rank[: min(self.K, self.pop_size)]
            g_old = normalized_tchebycheff_values(pop_F(self.pop[P]), self.z, self.znad, self.W[P])
            g_new = normalized_tchebycheff_values(np.repeat(off_f[None, :], len(P), axis=0), self.z, self.znad, self.W[P])
            repl = np.where(g_old >= g_new)[0][:1]
            if repl.size:
                self.pop[P[repl]] = off

    def _set_optimum(self):
        set_optimum_from_pop(self)

