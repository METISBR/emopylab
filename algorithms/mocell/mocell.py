# emopylab 2026
"""MOCell (cellular genetic algorithm) for EmoPyLab.

Reference:
A. J. Nebro, J. J. Durillo, F. Luna, B. Dorronsoro, and E. Alba. MOCell:
A cellular genetic algorithm for multiobjective optimization.
International Journal of Intelligent Systems, 2009, 24(7): 726-746.
"""

from __future__ import annotations

import math
import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from operators.sampling.rnd import (
    BinaryRandomSampling,
    FloatRandomSampling,
    IntegerRandomSampling,
    PermutationRandomSampling,
)
from util.optimum import filter_optimum

from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGAhalf import OperatorGAhalf
from operators.utility_functions.TournamentSelection import TournamentSelection


ALGORITHM_FLAGS = {
    "MOCell": {"multi", "real", "integer", "label", "binary", "permutation", "constrained"},
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


def _cv_values(pop: Population) -> np.ndarray:
    cv = pop.get("CV")
    if cv is not None:
        arr = np.asarray(cv, dtype=float).reshape(-1)
        if arr.size == len(pop):
            return np.maximum(arr, 0.0)
    cons = _constraint_matrix(pop)
    if cons.size == 0:
        return np.zeros(len(pop), dtype=float)
    return np.sum(np.maximum(cons, 0.0), axis=1)


def _build_moore_neighborhood(pop_size: int) -> np.ndarray:
    c_size = int(math.floor(math.sqrt(pop_size)))
    if c_size < 1:
        c_size = 1
    n_cells = c_size * c_size
    grid = np.arange(n_cells, dtype=int).reshape(c_size, c_size)
    shifts = [
        (0, 0), (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
    ]
    neigh = [np.roll(np.roll(grid, dr, axis=0), dc, axis=1).ravel() for dr, dc in shifts]
    return np.column_stack(neigh)


class MOCell(Algorithm):
    """MOCell local implementation compatible with EmoPyLab."""

    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.sampling = sampling
        self.neighbor = None
        self._cell_pop_size = None
        self.cell_archive = Population.empty()

    def _default_sampling(self):
        vtype = getattr(self.problem, "vtype", float)
        if vtype in (bool, np.bool_):
            return BinaryRandomSampling()
        if vtype in (int, np.int32, np.int64):
            return IntegerRandomSampling()
        # best-effort: if permutation-like bounds absent and variable type indicates int with unique domain,
        # users can inject custom sampling via framework config; default remains robust.
        return FloatRandomSampling()

    def _setup(self, problem, **kwargs):
        self.neighbor = _build_moore_neighborhood(self.pop_size)
        self._cell_pop_size = int(self.neighbor.shape[0])

    def _initialize_infill(self):
        sampling = self.sampling if self.sampling is not None else self._default_sampling()
        return sampling.do(self.problem, self._cell_pop_size, random_state=_rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.cell_archive = Population.empty()

    def _compute_rank_and_crowding(self):
        F = np.asarray(self.pop.get("F"), dtype=float)
        G = _constraint_matrix(self.pop)
        if G.size == 0:
            front_no, _ = NDSort(F, np.inf)
        else:
            front_no, _ = NDSort(F, G, np.inf)
        crowd = np.asarray(CrowdingDistance(F, front_no), dtype=float)
        return np.asarray(front_no, dtype=float), crowd

    def _infill(self):
        rng = _rng_from_algo(self)
        front_no, crowd = self._compute_rank_and_crowding()
        X = np.asarray(self.pop.get("X"))
        n, d = X.shape
        Xoff = np.empty((n, d), dtype=X.dtype if X.dtype != object else float)

        for i in range(n):
            nb = self.neighbor[i]
            parents_local = np.asarray(
                TournamentSelection(2, 2, front_no[nb], -crowd[nb], rng=rng),
                dtype=int,
            ).reshape(-1)
            parent_idx = nb[parents_local - 1]
            child = OperatorGAhalf(self.problem, X[parent_idx, :], rng=rng)
            Xoff[i, :] = np.asarray(child)[0, :]

        return Population.new("X", Xoff)

    def _advance(self, infills=None, **kwargs):
        rng = _rng_from_algo(self)
        pop = self.pop.copy()
        offspring = infills

        vio_p = _cv_values(pop)
        vio_o = _cv_values(offspring)
        Fp = np.asarray(pop.get("F"), dtype=float)
        Fo = np.asarray(offspring.get("F"), dtype=float)

        replace = (vio_o < vio_p) | ((vio_o == vio_p) & np.any(Fp > Fo, axis=1))
        if np.any(replace):
            idx = np.where(replace)[0]
            pop[idx] = offspring[idx]

        merged = Population.merge(self.cell_archive, offspring) if len(self.cell_archive) > 0 else offspring.copy()
        Fm = np.asarray(merged.get("F"), dtype=float)
        Gm = _constraint_matrix(merged)
        if Gm.size == 0:
            front_no, _ = NDSort(Fm, 1)
        else:
            front_no, _ = NDSort(Fm, Gm, 1)
        archive = merged[np.where(np.asarray(front_no) == 1)[0]]

        if len(archive) > self.pop_size:
            cd = np.asarray(CrowdingDistance(np.asarray(archive.get("F"), dtype=float), np.ones(len(archive))), dtype=float)
            rank = np.argsort(-cd)
            archive = archive[rank[: self.pop_size]]

        n_rep = min(20, len(pop), len(archive))
        if n_rep > 0:
            pop_idx = rng.permutation(len(pop))[:n_rep]
            arc_idx = rng.permutation(len(archive))[:n_rep]
            pop[pop_idx] = archive[arc_idx]

        self.pop = pop
        self.cell_archive = archive

    def _set_optimum(self):
        src = self.cell_archive if len(self.cell_archive) > 0 else self.pop
        self.opt = filter_optimum(src, least_infeasible=True)
