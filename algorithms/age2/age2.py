"""EmoPyLab Native AGE-MOEA-II (Approximation-Guided Evolutionary MOEA II)."""

from __future__ import annotations

from typing import Any, Optional, Union
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
from util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = [
    "AGEMOEA2",
]


class AGEMOEA2Survival(Survival):
    """Geometry/Approximation-guided environmental selection for AGE-MOEA-II."""

    def __init__(self, filter_infeasible: bool = True) -> None:
        super().__init__(filter_infeasible=filter_infeasible)
        self.nds = NonDominatedSorting()

    def _do(
        self,
        problem: Any,
        pop: Population,
        *args: Any,
        n_survive: int | None = None,
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
        # Select from last front based on diversity / approximation distance
        chosen_from_last = self._select_diverse(F, survivors, last_front, remaining)
        survivors.extend(chosen_from_last)

        return pop[np.array(survivors, dtype=int)]

    def _select_diverse(
        self,
        F: np.ndarray,
        survivor_indices: list[int],
        last_front: np.ndarray,
        n_needed: int,
    ) -> list[int]:
        if len(last_front) <= n_needed:
            return list(last_front)

        # Normalize objectives
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        span = np.maximum(f_max - f_min, 1e-12)
        F_norm = (F - f_min[None, :]) / span[None, :]

        chosen = []
        available = list(last_front)

        # Distance matrix from available to current selected
        selected_idx = list(survivor_indices)

        while len(chosen) < n_needed and len(available) > 0:
            if len(selected_idx) == 0:
                # Pick extreme or centroid
                pick = available.pop(0)
            else:
                F_avail = F_norm[available]
                F_sel = F_norm[selected_idx]

                # Pairwise distance: (n_avail, n_sel)
                dists = np.linalg.norm(F_avail[:, None, :] - F_sel[None, :, :], axis=2)
                min_dist_to_sel = np.min(dists, axis=1)

                # Pick one maximizing minimum distance (max-min diversity)
                best_rel_idx = int(np.argmax(min_dist_to_sel))
                pick = available.pop(best_rel_idx)

            chosen.append(pick)
            selected_idx.append(pick)

        return chosen


class AGEMOEA2(Algorithm):
    """Approximation-Guided Evolutionary Multi-Objective Algorithm II (AGE-MOEA-II)."""

    ALGO_FLAGS = {"multi", "many", "real", "integer", "constrained"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        pop_size: int = 100,
        sampling=None,
        selection=None,
        crossover=None,
        mutation=None,
        survival=None,
        eliminate_duplicates: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pop_size = int(max(2, pop_size))
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
        self.survival = survival or AGEMOEA2Survival()
        self.eliminate_duplicates = eliminate_duplicates
        self.dup_elim = DefaultDuplicateElimination() if eliminate_duplicates else None
        self.mating = Mating(
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=self.dup_elim,
        )
    def _setup(self, problem: Any, **kwargs: Any) -> None:
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
