"""EmoPyLab Native NSGA-III Solver and Reference Direction Survival."""

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
from operators.selection.tournament import TournamentSelection
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.TournamentSelection import TournamentSelection as UtilTournamentSelection
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.ref_dirs import get_reference_directions

__all__ = [
    "NSGA3",
    "ReferenceDirectionSurvival",
    "associate_to_niches",
]


def associate_to_niches(
    F: np.ndarray,
    ref_dirs: np.ndarray,
    ideal_point: np.ndarray | None = None,
    nadir_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associates objective vectors to reference directions/niches."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    N, M = F.shape

    ref_dirs = np.asarray(ref_dirs, dtype=float)
    if ref_dirs.ndim == 1:
        ref_dirs = ref_dirs.reshape(1, -1)

    if ideal_point is None:
        ideal_point = np.min(F, axis=0)
    else:
        ideal_point = np.asarray(ideal_point, dtype=float).reshape(-1)

    if nadir_point is None:
        nadir_point = np.max(F, axis=0)
    else:
        nadir_point = np.asarray(nadir_point, dtype=float).reshape(-1)

    # Shift and normalize
    F_shifted = F - ideal_point[None, :]
    span = np.maximum(nadir_point - ideal_point, 1e-12)
    F_norm = F_shifted / span[None, :]

    # Perpendicular distance to reference directions
    # Distance from point p to line through origin along ref_dir w:
    # d_perp = || p - (p . w / ||w||^2) w ||
    ref_norms = np.linalg.norm(ref_dirs, axis=1)
    ref_norms = np.maximum(ref_norms, 1e-12)
    w_unit = ref_dirs / ref_norms[:, None]

    # Projections: (N, n_ref)
    projections = F_norm @ w_unit.T  # (N, n_ref)
    
    # Distance matrix: (N, n_ref)
    # ||p||^2 - (p.w_unit)^2
    p_norms_sq = np.sum(F_norm ** 2, axis=1, keepdims=True)  # (N, 1)
    dist_sq = np.maximum(0.0, p_norms_sq - projections ** 2)
    distances = np.sqrt(dist_sq)

    closest_niche = np.argmin(distances, axis=1)
    closest_dist = np.min(distances, axis=1)

    return closest_niche, closest_dist, distances


class ReferenceDirectionSurvival(Survival):
    """Reference Direction-based environmental selection for NSGA-III."""

    def __init__(self, ref_dirs: np.ndarray, filter_infeasible: bool = True) -> None:
        super().__init__(filter_infeasible=filter_infeasible)
        self.ref_dirs = np.asarray(ref_dirs, dtype=float)
        self.nds = NonDominatedSorting()

    def _do(
        self,
        problem: Any,
        pop: Population,
        *args: Any,
        n_survive: int | None = None,
        random_state=None,
        **kwargs: Any,
    ) -> Population:
        if n_survive is None or len(pop) <= n_survive:
            return pop

        F = np.asarray(pop.get("F"), dtype=float)
        N, M = F.shape

        fronts = self.nds.do(F)
        survivors = []
        last_front = None

        for front in fronts:
            if len(survivors) + len(front) <= n_survive:
                survivors.extend(front)
            else:
                last_front = front
                break

        if len(survivors) == n_survive or last_front is None:
            return pop[np.array(survivors, dtype=int)]

        remaining = n_survive - len(survivors)
        chosen_from_last = self._select_from_last_front(
            F, survivors, last_front, remaining, random_state=random_state
        )
        survivors.extend(chosen_from_last)
        return pop[np.array(survivors, dtype=int)]

    def _select_from_last_front(
        self,
        F: np.ndarray,
        survivor_indices: list[int],
        last_front: np.ndarray,
        n_needed: int,
        random_state=None,
    ) -> list[int]:
        rng = random_state if random_state is not None else np.random.default_rng()
        all_candidates = list(survivor_indices) + list(last_front)
        F_sub = F[all_candidates]
        ideal = np.min(F_sub, axis=0)
        nadir = np.max(F_sub, axis=0)

        niche_of_ind, dist_to_niche, _ = associate_to_niches(F, self.ref_dirs, ideal, nadir)

        # Count niche counts in current survivors
        n_ref = len(self.ref_dirs)
        niche_counts = np.zeros(n_ref, dtype=int)
        for idx in survivor_indices:
            niche_counts[niche_of_ind[idx]] += 1

        chosen = []
        available_last = list(last_front)

        while len(chosen) < n_needed and len(available_last) > 0:
            min_count = np.min(niche_counts)
            candidate_niches = np.where(niche_counts == min_count)[0]
            target_niche = int(rng.choice(candidate_niches))

            # Find individuals in available_last associated with target_niche
            matching = [idx for idx in available_last if niche_of_ind[idx] == target_niche]

            if len(matching) == 0:
                niche_counts[target_niche] = 1000000000
                continue

            if niche_counts[target_niche] == 0:
                # Pick individual with minimum distance
                dists = [dist_to_niche[idx] for idx in matching]
                best_idx = matching[int(np.argmin(dists))]
            else:
                best_idx = int(rng.choice(matching))

            chosen.append(best_idx)
            available_last.remove(best_idx)
            niche_counts[target_niche] += 1

        # Fallback if any remaining
        while len(chosen) < n_needed and len(available_last) > 0:
            pick = available_last.pop(0)
            chosen.append(pick)

        return chosen


class NSGA3(Algorithm):
    """Reference-Point Based Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    ALGO_FLAGS = {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        ref_dirs: Optional[np.ndarray] = None,
        pop_size: int | None = None,
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
    def _setup(self, problem: Any, **kwargs: Any) -> None:
        if self.ref_dirs is None:
            n_obj = getattr(problem, "n_obj", 3)
            self.ref_dirs = get_reference_directions("das-dennis", n_obj=n_obj, n_partitions=12)
        else:
            self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)

        if self.pop_size is None:
            self.pop_size = len(self.ref_dirs)
        self.pop_size = int(max(2, self.pop_size))

        if self.survival is None:
            self.survival = ReferenceDirectionSurvival(self.ref_dirs)

        if self.selection is None:
            from operators.selection.tournament import TournamentSelection
            from algorithms.moo.sms import cv_and_dom_tournament
            self.selection = TournamentSelection(func_comp=cv_and_dom_tournament)

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
        merged = Population.merge(self.pop, infills)
        self.pop = self.survival.do(self.problem, merged, n_survive=self.pop_size)
        self._set_optimum()
