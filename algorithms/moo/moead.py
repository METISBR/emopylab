"""EmoPyLab Native MOEA/D Solver."""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from operators.crossover.sbx import SBX
from operators.mutation.pm import PolynomialMutation
from operators.sampling.rnd import FloatRandomSampling
from util.ref_dirs import get_reference_directions

__all__ = [
    "MOEAD",
]


class MOEAD(Algorithm):
    """Multi-Objective Evolutionary Algorithm Based on Decomposition (MOEA/D)."""

    ALGO_FLAGS = {"multi", "many", "real", "integer", "constrained"}
    OBJECTIVE_SCOPE = "multi"

    def __init__(
        self,
        ref_dirs: Optional[np.ndarray] = None,
        n_neighbors: int = 20,
        prob_neighbor_mating: float = 0.9,
        sampling=None,
        crossover=None,
        mutation=None,
        decomposition: str = "tchebi",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ref_dirs = ref_dirs
        self.n_neighbors = int(n_neighbors)
        self.prob_neighbor_mating = float(prob_neighbor_mating)
        self.sampling = sampling or FloatRandomSampling()
        self.crossover = crossover or SBX(prob=1.0, eta=20)
        self.mutation = mutation or PolynomialMutation(eta=20)
        self.decomposition = decomposition

        self.W: Optional[np.ndarray] = None
        self.neighbors: Optional[np.ndarray] = None
        self.ideal_point: Optional[np.ndarray] = None
        self._current_subproblem: int = 0

    def _setup(self, problem: Any, **kwargs: Any) -> None:
        if self.ref_dirs is None:
            n_obj = getattr(problem, "n_obj", 3)
            self.ref_dirs = get_reference_directions("das-dennis", n_obj=n_obj, n_partitions=12)
        else:
            self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)

        self.W = np.asarray(self.ref_dirs, dtype=float)
        self.pop_size = len(self.W)

        # Distance between weight vectors to form neighborhood
        dist_W = np.linalg.norm(self.W[:, None, :] - self.W[None, :, :], axis=2)
        k_neighbors = min(self.n_neighbors, self.pop_size)
        self.neighbors = np.argsort(dist_W, axis=1)[:, :k_neighbors]

    def _initialize_infill(self) -> Population:
        return self.sampling.do(self.problem, self.pop_size, random_state=self.random_state)

    def _initialize_advance(self, infills: Population = None, **kwargs: Any) -> None:
        self.pop = infills if infills is not None else Population.empty()
        F = np.asarray(self.pop.get("F"), dtype=float)
        self.ideal_point = np.min(F, axis=0)
        self._set_optimum()

    def _infill(self) -> Population:
        if self.pop is None or len(self.pop) == 0:
            return self.sampling.do(self.problem, self.pop_size, random_state=self.random_state)

        rng = self.random_state if self.random_state is not None else np.random.default_rng()
        parents_pairs = []
        for i in range(self.pop_size):
            if rng.random() < self.prob_neighbor_mating:
                pool = self.neighbors[i]
            else:
                pool = np.arange(self.pop_size)

            parents_idx = rng.choice(pool, size=2, replace=False)
            parents_pairs.append([self.pop[parents_idx[0]], self.pop[parents_idx[1]]])

        off = self.crossover.do(self.problem, parents_pairs, random_state=rng)
        off = self.mutation.do(self.problem, off, random_state=rng)
        return off[:self.pop_size]

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            return

        F_off = np.asarray(infills.get("F"), dtype=float)
        X_off = np.asarray(infills.get("X"), dtype=float)

        self.ideal_point = np.minimum(self.ideal_point, np.min(F_off, axis=0))

        rng = self.random_state if self.random_state is not None else np.random.default_rng()

        for i in range(min(len(infills), self.pop_size)):
            off_ind = infills[i]
            f_child = F_off[i]

            if rng.random() < self.prob_neighbor_mating:
                indices = self.neighbors[i]
            else:
                indices = np.arange(self.pop_size)

            W_sub = self.W[indices]
            F_curr = np.asarray(self.pop[indices].get("F"), dtype=float)

            # Tchebycheff aggregation
            g_old = np.max(W_sub * np.abs(F_curr - self.ideal_point[None, :]), axis=1)
            g_new = np.max(W_sub * np.abs(f_child[None, :] - self.ideal_point[None, :]), axis=1)

            # Replace worse solutions
            replace_mask = g_new < g_old
            for rep_idx in indices[replace_mask]:
                self.pop[rep_idx] = off_ind

        self._set_optimum()
