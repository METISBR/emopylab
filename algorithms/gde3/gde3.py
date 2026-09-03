# emopylab 2026
"""GDE3 (Generalized Differential Evolution 3) for EmoPyLab.

Reference:
S. Kukkonen and J. Lampinen. GDE3: The third evolution step of generalized
differential evolution. Proceedings of the IEEE Congress on Evolutionary
Computation, 2005, 443-450.
"""

from __future__ import annotations

import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from operators.sampling.rnd import FloatRandomSampling, IntegerRandomSampling
from util.optimum import filter_optimum

from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorDE import OperatorDE


ALGORITHM_FLAGS = {
    "GDE3": {"multi", "real", "integer", "constrained"},
}


def _rng_from_algo(algo: Algorithm) -> np.random.Generator:
    rng = getattr(algo, "random_state", None)
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        rng = np.random.default_rng()
        setattr(algo, "random_state", rng)
        return rng
    return np.random.default_rng(int(rng))


def _constraint_matrix(pop: Population) -> np.ndarray:
    g = pop.get("G")
    if g is None:
        return np.zeros((len(pop), 0), dtype=float)
    g = np.asarray(g, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    return g


def _is_feasible(cons: np.ndarray) -> np.ndarray:
    if cons.size == 0:
        return np.ones(cons.shape[0], dtype=bool)
    return np.all(cons <= 0.0, axis=1)


def _environmental_selection(population: Population, offspring: Population, n_survive: int) -> Population:
    pop_obj = np.asarray(population.get("F"), dtype=float)
    pop_con = _constraint_matrix(population)
    feasible_p = _is_feasible(pop_con)

    off_obj = np.asarray(offspring.get("F"), dtype=float)
    off_con = _constraint_matrix(offspring)
    feasible_o = _is_feasible(off_con)

    if pop_con.size == 0:
        weakly_better_con = np.ones(len(population), dtype=bool)
    else:
        weakly_better_con = np.all(pop_con >= off_con, axis=1)

    updated = (
        (~feasible_p & feasible_o)
        | (~feasible_p & ~feasible_o & weakly_better_con)
        | (feasible_p & feasible_o & np.all(pop_obj >= off_obj, axis=1))
    )

    selected = (
        feasible_p
        & feasible_o
        & np.any(pop_obj < off_obj, axis=1)
        & np.any(pop_obj > off_obj, axis=1)
    )

    population = population.copy()
    if np.any(updated):
        population[np.where(updated)[0]] = offspring[np.where(updated)[0]]

    if np.any(selected):
        population = Population.merge(population, offspring[np.where(selected)[0]])

    pop_obj = np.asarray(population.get("F"), dtype=float)
    pop_con = _constraint_matrix(population)
    feasible = _is_feasible(pop_con)

    front_no = np.full(len(population), np.inf, dtype=float)
    max_f_no = 0
    if np.any(feasible):
        front_feas, max_f_no = NDSort(pop_obj[feasible, :], np.inf)
        front_no[np.where(feasible)[0]] = front_feas
    if np.any(~feasible):
        front_inf, _ = NDSort(pop_con[~feasible, :], np.inf)
        front_no[np.where(~feasible)[0]] = front_inf + max_f_no

    finite_fronts = front_no[np.isfinite(front_no)].astype(int)
    if finite_fronts.size == 0:
        return population[: min(n_survive, len(population))]
    hist = np.bincount(finite_fronts, minlength=int(np.max(finite_fronts)) + 1)
    csum = np.cumsum(hist[1:])
    max_keep_front = int(np.where(csum >= n_survive)[0][0] + 1)
    last_front = np.where(front_no == float(max_keep_front))[0].tolist()
    already_kept = int(np.sum(front_no < float(max_keep_front)))

    while len(last_front) > max(0, n_survive - already_kept):
        cd = CrowdingDistance(pop_obj[last_front, :])
        worst_local = int(np.argmin(np.asarray(cd, dtype=float)))
        last_front.pop(worst_local)

    keep = np.concatenate([np.where(front_no < float(max_keep_front))[0], np.asarray(last_front, dtype=int)])
    return population[keep]


class GDE3(Algorithm):
    """GDE3 local implementation compatible with EmoPyLab."""

    def __init__(
        self,
        pop_size: int = 100,
        CR: float = 1.0,
        F: float = 0.5,
        proM: float = 1.0,
        disM: float = 20.0,
        sampling=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.CR = float(CR)
        self.F = float(F)
        self.proM = float(proM)
        self.disM = float(disM)
        self.sampling = sampling

    def _default_sampling(self):
        vtype = getattr(self.problem, "vtype", float)
        if vtype in (int, np.int32, np.int64):
            return IntegerRandomSampling()
        return FloatRandomSampling()

    def _initialize_infill(self):
        sampling = self.sampling if self.sampling is not None else self._default_sampling()
        return sampling.do(self.problem, self.pop_size, random_state=_rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills

    def _infill(self):
        rng = _rng_from_algo(self)
        n = len(self.pop)
        idx2 = rng.integers(0, n, size=n)
        idx3 = rng.integers(0, n, size=n)

        X1 = self.pop.get("X")
        X2 = X1[idx2]
        X3 = X1[idx3]
        Xoff = OperatorDE(
            self.problem,
            X1,
            X2,
            X3,
            Parameter=[self.CR, self.F, self.proM, self.disM],
            rng=rng,
        )
        return Population.new("X", np.asarray(Xoff, dtype=float))

    def _advance(self, infills=None, **kwargs):
        self.pop = _environmental_selection(self.pop, infills, self.pop_size)

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)

