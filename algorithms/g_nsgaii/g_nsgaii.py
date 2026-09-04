# emopylab 2026
"""g-NSGA-II (g-dominance based NSGA-II).

Reference:
J. Molina et al. EJOR, 2009, 197(2): 685-692.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from util.optimum import filter_optimum

from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection
from algorithms.community_utils.moead_family import rng_from_algo, sample_initial


ALGORITHM_FLAGS = {"gNSGAII": {"multi", "real", "integer", "label", "binary", "permutation"}}


def _evaluate_g(F: np.ndarray, point: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    P = np.repeat(np.asarray(point, dtype=float)[None, :], F.shape[0], axis=0)
    flag = np.all(F <= P, axis=1) | np.all(F >= P, axis=1)
    out = F.copy()
    out[~flag, :] = out[~flag, :] + 1e10
    return out


def _env_selection(pop: Population, N: int, point: np.ndarray):
    evalF = _evaluate_g(pop.get("F"), point)
    front, maxf = NDSort(evalF, N)
    front = np.asarray(front, dtype=float)
    next_mask = front < maxf
    crowd = np.asarray(CrowdingDistance(evalF, front), dtype=float)
    last = np.where(front == float(maxf))[0]
    need = int(N - np.sum(next_mask))
    if need > 0 and last.size:
        rank = np.argsort(-crowd[last])
        next_mask[last[rank[:need]]] = True
    idx = np.where(next_mask)[0]
    return pop[idx], front[idx], crowd[idx]


class gNSGAII(Algorithm):
    def __init__(self, pop_size: int = 100, Point=None, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.Point = Point
        self.sampling = sampling

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        if self.Point is None:
            self.Point = np.zeros(self.problem.n_obj, dtype=float) + 0.5
        evalF = _evaluate_g(self.pop.get("F"), self.Point)
        self.front_no = np.asarray(NDSort(evalF, np.inf)[0], dtype=float)
        self.crowd = np.asarray(CrowdingDistance(evalF, self.front_no), dtype=float)

    def _infill(self):
        rng = rng_from_algo(self)
        mating = np.asarray(TournamentSelection(2, self.pop_size, self.front_no, -self.crowd, rng=rng), dtype=int) - 1
        return OperatorGA(self.problem, self.pop[mating], rng=rng)

    def _advance(self, infills=None, **kwargs):
        self.pop, self.front_no, self.crowd = _env_selection(Population.merge(self.pop, infills), self.pop_size, self.Point)

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)

