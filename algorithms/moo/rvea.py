"""EmoPyLab Native RVEA (Reference Vector Guided Evolutionary Algorithm)."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from core.algorithm import Algorithm
from core.duplicate import DefaultDuplicateElimination
from core.mating import Mating
from core.population import Population
from core.survival import Survival
from operators.crossover.sbx import SBX
from operators.mutation.pm import PolynomialMutation
from operators.sampling.rnd import FloatRandomSampling
from util.ref_dirs import get_reference_directions

__all__ = [
    "RVEA",
]


def _cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes cosine similarity between sets of vectors A (N, M) and B (K, M)."""
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    norm_A = np.maximum(norm_A, 1e-12)
    norm_B = np.maximum(norm_B, 1e-12)
    return np.clip((A @ B.T) / (norm_A @ norm_B.T), -1.0, 1.0)


class RVEASurvival(Survival):
    """Angle-penalized distance (APD) based environmental selection for RVEA."""

    def __init__(self, ref_dirs: np.ndarray, alpha: float = 2.0) -> None:
        super().__init__(filter_infeasible=True)
        self.ref_dirs = np.asarray(ref_dirs, dtype=float)
        self.alpha = float(alpha)

        # Precompute acute angles between reference vectors
        cos_vv = _cosine_similarity_matrix(self.ref_dirs, self.ref_dirs)
        np.fill_diagonal(cos_vv, 0.0)
        self.gamma = np.maximum(np.min(np.arccos(np.clip(cos_vv, -1.0, 1.0)), axis=1), 1e-12)

    def _do(
        self,
        problem: Any,
        pop: Population,
        *args: Any,
        n_survive: int | None = None,
        theta: float = 0.0,
        z_min: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Population:
        if n_survive is None or len(pop) <= n_survive:
            return pop

        F = np.asarray(pop.get("F"), dtype=float)
        N, M = F.shape

        if z_min is None:
            z_min = np.min(F, axis=0)

        # Objective translation
        F_trans = F - z_min[None, :]

        # Calculate angles between translated objective vectors and reference vectors
        cos_angles = _cosine_similarity_matrix(F_trans, self.ref_dirs)
        angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

        # Associate each individual with its nearest reference vector
        associations = np.argmin(angles, axis=1)

        survivor_indices = []

        for k in range(len(self.ref_dirs)):
            ind_idx = np.where(associations == k)[0]
            if len(ind_idx) == 0:
                continue

            # APD calculation: (1 + M * (t/t_max)^alpha * (theta_k / gamma_k)) * ||F_trans||
            sub_angles = angles[ind_idx, k]
            f_norms = np.linalg.norm(F_trans[ind_idx], axis=1)

            apd = (1.0 + M * (theta ** self.alpha) * (sub_angles / self.gamma[k])) * f_norms
            best_ind = ind_idx[np.argmin(apd)]
            survivor_indices.append(int(best_ind))

        # If more or fewer than n_survive:
        if len(survivor_indices) > n_survive:
            survivor_indices = survivor_indices[:n_survive]
        elif len(survivor_indices) < n_survive:
            remaining = list(set(range(len(pop))) - set(survivor_indices))
            if remaining:
                needed = n_survive - len(survivor_indices)
                survivor_indices.extend(remaining[:needed])

        return pop[np.array(survivor_indices, dtype=int)]


class RVEA(Algorithm):
    """Reference Vector Guided Evolutionary Algorithm (RVEA) in pure EmoPyLab."""

    ALGO_FLAGS = {"multi", "many", "real", "integer", "constrained"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        ref_dirs: Optional[np.ndarray] = None,
        pop_size: int | None = None,
        alpha: float = 2.0,
        sampling=None,
        selection=None,
        crossover=None,
        mutation=None,
        survival=None,
        eliminate_duplicates: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ref_dirs = ref_dirs
        self.pop_size = pop_size
        self.alpha = float(alpha)
        if sampling is None:
            self.sampling = FloatRandomSampling()
        elif isinstance(sampling, (np.ndarray, Population, list)):
            from core.sampling import Sampling
            class _StaticSampling(Sampling):
                def __init__(self, init_data):
                    super().__init__()
                    self.init_data = init_data
                def _do(self, problem, n_samples, **kwargs):
                    if isinstance(self.init_data, Population):
                        return self.init_data[:n_samples]
                    arr = np.asarray(self.init_data)
                    return Population.new("X", arr[:n_samples])
            self.sampling = _StaticSampling(sampling)
        else:
            self.sampling = sampling
        from operators.selection.tournament import TournamentSelection
        from algorithms.moo.sms import cv_and_dom_tournament
        self.selection = selection if selection is not None else TournamentSelection(func_comp=cv_and_dom_tournament)
        self.crossover = crossover or SBX(prob=0.9, eta=15)
        self.mutation = mutation or PolynomialMutation(eta=20)
        self.survival = survival
        self.eliminate_duplicates = eliminate_duplicates
        self.dup_elim = DefaultDuplicateElimination() if eliminate_duplicates else None
        self.mating = Mating(
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=self.dup_elim,
        )
        self.z_min: Optional[np.ndarray] = None
        self._max_iter_estimate: int = 250

    def _setup(self, problem: Any, **kwargs: Any) -> None:
        if self.ref_dirs is None:
            n_obj = getattr(problem, "n_obj", 3)
            self.ref_dirs = get_reference_directions("das-dennis", n_obj=n_obj, n_partitions=12)
        else:
            self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)

        if self.pop_size is None:
            self.pop_size = len(self.ref_dirs)
        self.pop_size = int(max(2, self.pop_size))

        if self.selection is None:
            from operators.selection.tournament import TournamentSelection
            from algorithms.moo.sms import cv_and_dom_tournament
            self.selection = TournamentSelection(func_comp=cv_and_dom_tournament)

        if self.survival is None:
            self.survival = RVEASurvival(self.ref_dirs, alpha=self.alpha)

        self.mating = Mating(
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=self.dup_elim,
        )

    def _initialize_infill(self) -> Population:
        return self.sampling.do(self.problem, self.pop_size, random_state=self.random_state)

    def _initialize_advance(self, infills: Population = None, **kwargs: Any) -> None:
        self.pop = infills if infills is not None else Population.empty()
        F = np.asarray(self.pop.get("F"), dtype=float)
        self.z_min = np.min(F, axis=0)
        self._set_optimum()

    def _infill(self) -> Population:
        if self.pop is None or len(self.pop) == 0:
            return self.sampling.do(self.problem, self.pop_size, random_state=self.random_state)
        return self.mating.do(
            self.problem,
            self.pop,
            self.pop_size,
            random_state=self.random_state,
        )

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            return

        F_off = np.asarray(infills.get("F"), dtype=float)
        self.z_min = np.minimum(self.z_min, np.min(F_off, axis=0))

        merged = Population.merge(self.pop, infills)
        
        # Calculate search progress theta in [0, 1]
        n_iter = self.n_iter or 1
        n_max = self.termination.n_max_gen if hasattr(self.termination, "n_max_gen") else self._max_iter_estimate
        theta = min(1.0, float(n_iter) / float(max(1, n_max)))

        self.pop = self.survival.do(
            self.problem,
            merged,
            n_survive=self.pop_size,
            theta=theta,
            z_min=self.z_min,
        )
        self._set_optimum()
