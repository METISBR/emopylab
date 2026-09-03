# emopylab 2026
"""e-MOEA (epsilon multi-objective evolutionary algorithm).

Reference:
K. Deb, M. Mohan, and S. Mishra. EMO 2003, 222-236.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from util.optimum import filter_optimum

from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGAhalf import OperatorGAhalf
from algorithms.community_utils.moead_family import pop_F, rng_from_algo, sample_initial


ALGORITHM_FLAGS = {"eMOEA": {"multi", "many", "real", "integer", "label", "binary", "permutation"}}


def _grid_locations(F: np.ndarray, epsilon: float):
    F = np.asarray(F, dtype=float)
    fmin = np.min(F, axis=0)
    return np.floor((F - fmin) / epsilon), fmin


def _ind_f(ind) -> np.ndarray:
    return np.asarray(ind.get("F") if hasattr(ind, "get") else ind.F, dtype=float).reshape(-1)


def _update_population(pop: Population, offspring, rng) -> Population:
    N = len(pop)
    F = pop_F(pop)
    off_f = _ind_f(offspring)
    if np.any(np.all(F <= off_f[None, :], axis=1)):
        return pop

    dominated = np.where(np.all(off_f[None, :] <= F, axis=1))[0]
    pop = pop.copy()
    if dominated.size:
        pop[int(dominated[rng.integers(0, dominated.size)])] = offspring
    else:
        pop[int(rng.integers(0, N))] = offspring
    return pop


def _update_archive(archive: Population, offspring, epsilon: float) -> Population:
    if len(archive) == 0:
        return Population.create(offspring)

    Aobj = pop_F(archive)
    off_f = _ind_f(offspring)
    child_grid = np.floor((off_f - np.min(Aobj, axis=0)) / epsilon)
    arch_grid = np.floor((Aobj - np.min(Aobj, axis=0)) / epsilon)

    if np.any(np.all(arch_grid <= child_grid[None, :], axis=1)):
        return archive

    dominate = np.where(np.all(child_grid[None, :] <= arch_grid, axis=1))[0]
    if dominate.size:
        keep = np.ones(len(archive), dtype=bool)
        keep[dominate] = False
        return Population.merge(archive[np.where(keep)[0]], Population.create(offspring))

    same = np.where(np.all(arch_grid == child_grid[None, :], axis=1))[0]
    if same.size:
        B = child_grid * epsilon + np.min(Aobj, axis=0)
        if np.linalg.norm(off_f - B) < np.linalg.norm(Aobj[int(same[0])] - B):
            archive = archive.copy()
            archive[int(same[0])] = offspring
        return archive

    return Population.merge(archive, Population.create(offspring))


class eMOEA(Algorithm):
    def __init__(self, pop_size: int = 100, epsilon: float = 0.06, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.epsilon = float(epsilon)
        self.sampling = sampling
        self.e_archive = Population.empty()

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        grids, _ = _grid_locations(pop_F(self.pop), self.epsilon)
        front = np.asarray(NDSort(grids, 1)[0], dtype=float)
        self.e_archive = self.pop[np.where(front == 1)[0]]

    def _infill(self):
        rng = rng_from_algo(self)
        offs = []
        for _ in range(self.pop_size):
            k = rng.permutation(self.pop_size)[:2]
            f1 = _ind_f(self.pop[int(k[0])])
            f2 = _ind_f(self.pop[int(k[1])])
            domi = int(np.any(f1 < f2)) - int(np.any(f1 > f2))
            p = int(k[(domi == -1) + 0])
            if len(self.e_archive) > 0:
                q = int(rng.integers(0, len(self.e_archive)))
                pair = Population.create(self.pop[p], self.e_archive[q])
            else:
                q = int(rng.integers(0, self.pop_size))
                pair = Population.create(self.pop[p], self.pop[q])
            off = OperatorGAhalf(self.problem, pair, rng=rng)
            offs.append(off[0] if isinstance(off, Population) else off)
        out = Population.empty()
        for off in offs:
            out = Population.merge(out, Population.create(off))
        return out

    def _advance(self, infills=None, **kwargs):
        rng = rng_from_algo(self)
        for off in infills:
            self.pop = _update_population(self.pop, off, rng)
            self.e_archive = _update_archive(self.e_archive, off, self.epsilon)

    def _set_optimum(self):
        self.opt = filter_optimum(self.e_archive if len(self.e_archive) > 0 else self.pop, least_infeasible=True)
