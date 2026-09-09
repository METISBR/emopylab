"""EmoPyLab SMS-EMOA and tournament selection helpers."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from core.survival import Survival
from operators.crossover.sbx import SBX
from operators.mutation.pm import PolynomialMutation
from operators.sampling.rnd import FloatRandomSampling
from operators.selection.tournament import TournamentSelection, compare
from util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = [
    "cv_and_dom_tournament",
    "SMSEMOA",
]


def cv_and_dom_tournament(pop: Population, P: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Tournament selection function comparing constraint violation (CV) and dominance."""
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
            if rank_a is not None and rank_b is not None and rank_a != rank_b:
                S[i] = a if rank_a < rank_b else b
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


class SMSEMOASurvival(Survival):
    """Survival operator for SMS-EMOA."""

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem: Any, pop: Population, n_survive: int = 100, **kwargs: Any) -> Population:
        if len(pop) <= n_survive:
            return pop

        F = np.asarray(pop.get("F"), dtype=float)
        fronts = self.nds.do(F)

        survivors = []
        for front in fronts:
            if len(survivors) + len(front) <= n_survive:
                survivors.extend(front)
            else:
                remaining = n_survive - len(survivors)
                survivors.extend(front[:remaining])
                break

        return pop[np.array(survivors, dtype=int)]


class SMSEMOA(Algorithm):
    """S-Metric Selection Evolutionary Multi-Objective Algorithm (SMS-EMOA)."""

    def __init__(
        self,
        pop_size: int = 100,
        sampling=None,
        selection=None,
        crossover=None,
        mutation=None,
        survival=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.sampling = sampling or FloatRandomSampling()
        self.selection = selection or TournamentSelection(func_comp=cv_and_dom_tournament)
        self.crossover = crossover or SBX(prob=0.9, eta=15)
        self.mutation = mutation or PolynomialMutation(eta=20)
        self.survival = survival or SMSEMOASurvival()

    def _initialize_infill(self) -> Population:
        return self.sampling.do(self.problem, self.pop_size, random_state=self.random_state)

    def _initialize_advance(self, infills: Population = None, **kwargs: Any) -> None:
        self.pop = infills
        self._set_optimum()

    def _infill(self) -> Population:
        return self.crossover.do(
            self.problem,
            self.pop,
            random_state=self.random_state,
        )

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            return
        merged = Population.merge(self.pop, infills)
        self.pop = self.survival.do(self.problem, merged, n_survive=self.pop_size)
        self._set_optimum()
