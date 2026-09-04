"""EmoPyLab Native NSGA-II Solver and binary tournament operator."""

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
from operators.selection.tournament import TournamentSelection, compare
from operators.survival.rank_and_crowding.classes import RankAndCrowding
from util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = [
    "NSGA2",
    "binary_tournament",
]


def binary_tournament(pop: Population, P: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Binary tournament comparison function based on dominance rank and crowding distance."""
    n_tournaments, n_parents = P.shape
    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)
    random_state = kwargs.get("random_state", None)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        ind_a, ind_b = pop[a], pop[b]

        cv_a = float(getattr(ind_a, "cv", 0.0) or 0.0)
        cv_b = float(getattr(ind_b, "cv", 0.0) or 0.0)

        if cv_a != cv_b:
            S[i] = a if cv_a < cv_b else b
        else:
            rank_a = getattr(ind_a, "rank", None)
            rank_b = getattr(ind_b, "rank", None)
            crowd_a = getattr(ind_a, "crowding", None)
            crowd_b = getattr(ind_b, "crowding", None)

            if rank_a is not None and rank_b is not None and rank_a != rank_b:
                S[i] = a if rank_a < rank_b else b
            elif crowd_a is not None and crowd_b is not None and crowd_a != crowd_b:
                S[i] = a if crowd_a > crowd_b else b
            else:
                f_a = getattr(ind_a, "F", None)
                f_b = getattr(ind_b, "F", None)
                if f_a is not None and f_b is not None:
                    f_a = np.asarray(f_a, dtype=float).reshape(-1)
                    f_b = np.asarray(f_b, dtype=float).reshape(-1)
                    a_dom_b = np.all(f_a <= f_b) and np.any(f_a < f_b)
                    b_dom_a = np.all(f_b <= f_a) and np.any(f_b < f_a)
                    if a_dom_b and not b_dom_a:
                        S[i] = a
                    elif b_dom_a and not a_dom_b:
                        S[i] = b
                    else:
                        if random_state is not None:
                            S[i] = random_state.choice([a, b])
                        else:
                            S[i] = np.random.choice([a, b])
                else:
                    if random_state is not None:
                        S[i] = random_state.choice([a, b])
                    else:
                        S[i] = np.random.choice([a, b])

    return S[:, None].astype(int, copy=False)


class NSGA2(Algorithm):
    """Non-dominated Sorting Genetic Algorithm II (NSGA-II) in pure EmoPyLab."""

    ALGO_FLAGS = {"multi", "real", "integer", "binary", "permutation", "label", "constrained"}
    OBJECTIVE_SCOPE = "multi"

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
        self.selection = selection if selection is not None else TournamentSelection(func_comp=binary_tournament)
        self.crossover = crossover or SBX(prob=0.9, eta=15)
        self.mutation = mutation or PolynomialMutation(eta=20)
        self.survival = survival or RankAndCrowding()
        self.eliminate_duplicates = eliminate_duplicates
        self.dup_elim = DefaultDuplicateElimination() if eliminate_duplicates else None

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
        self._update_ranks_and_crowding(self.pop)
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
        self.pop = self.survival.do(
            self.problem,
            merged,
            n_survive=self.pop_size,
            algorithm=self,
            random_state=self.random_state,
            **kwargs,
        )
        if hasattr(self.survival, "diagnostics"):
            self.diagnostics = getattr(self.survival, "diagnostics", {})
        self._update_ranks_and_crowding(self.pop)
        self._set_optimum()

    def _update_ranks_and_crowding(self, pop: Population) -> None:
        if pop is None or len(pop) == 0:
            return
        try:
            F = np.asarray(pop.get("F"), dtype=float)
            nds = NonDominatedSorting()
            fronts = nds.do(F)
            for rank, front in enumerate(fronts):
                for idx in front:
                    pop[idx].set("rank", rank)
        except Exception:
            pass
