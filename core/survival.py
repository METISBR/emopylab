"""
Standalone implementation of Survival, ToReplacement, and split_by_feasibility (EmoPyLab 2026).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from core.operator import default_random_state
from core.population import Population

__all__ = [
    "Survival",
    "ToReplacement",
    "split_by_feasibility",
]


class Survival:
    """Base class for survival selection operators."""

    def __init__(self, filter_infeasible: bool = True) -> None:
        super().__init__()
        self.filter_infeasible = filter_infeasible

    @default_random_state
    def do(
        self,
        problem,
        pop: Population,
        *args: Any,
        n_survive: Optional[int] = None,
        random_state=None,
        return_indices: bool = False,
        **kwargs: Any,
    ) -> Union[Population, List[int]]:
        if len(pop) == 0:
            return [] if return_indices else pop

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        if self.filter_infeasible and problem.has_constraints():
            feas, infeas = split_by_feasibility(pop, sort_infeas_by_cv=True)

            if len(feas) == 0:
                survivors = Population()
            else:
                survivors = self._do(
                    problem,
                    pop[feas],
                    *args,
                    n_survive=min(len(feas), n_survive),
                    random_state=random_state,
                    **kwargs,
                )

            n_remaining = n_survive - len(survivors)
            if n_remaining > 0 and len(infeas) > 0:
                survivors = Population.merge(survivors, pop[infeas[:n_remaining]])
        else:
            survivors = self._do(
                problem,
                pop,
                *args,
                n_survive=n_survive,
                random_state=random_state,
                **kwargs,
            )

        if return_indices:
            H = {ind: k for k, ind in enumerate(pop)}
            return [H[survivor] for survivor in survivors if survivor in H]
        else:
            return survivors

    @abstractmethod
    def _do(
        self,
        problem,
        pop: Population,
        *args: Any,
        n_survive: Optional[int] = None,
        random_state=None,
        **kwargs: Any,
    ) -> Population:
        pass


class ToReplacement(Survival):
    """Wraps a survival operator into a replacement operator."""

    def __init__(self, survival: Survival) -> None:
        super().__init__(filter_infeasible=False)
        self.survival = survival

    def _do(
        self,
        problem,
        pop: Population,
        off: Population,
        random_state=None,
        **kwargs: Any,
    ) -> Population:
        merged = Population.merge(pop, off)
        I = self.survival.do(
            problem,
            merged,
            n_survive=len(merged),
            return_indices=True,
            random_state=random_state,
            **kwargs,
        )
        merged.set("__rank__", I)

        for k in range(len(pop)):
            if off[k].get("__rank__") < pop[k].get("__rank__"):
                pop[k] = off[k]

        return pop


def split_by_feasibility(
    pop: Population,
    sort_infeas_by_cv: bool = True,
    sort_feas_by_obj: bool = False,
    return_pop: bool = False,
):
    """Split population into feasible and infeasible indices or subpopulations."""
    F, CV, b = pop.get("F", "CV", "FEAS")

    b = np.asarray(b).astype(bool)
    if b.ndim > 1:
        b = b[:, 0]

    feasible = np.where(b)[0]
    infeasible = np.where(~b)[0]

    if sort_infeas_by_cv and len(infeasible) > 0:
        cv_arr = np.asarray(CV)
        if cv_arr.ndim > 1:
            cv_arr = cv_arr[:, 0]
        infeasible = infeasible[np.argsort(cv_arr[infeasible])]

    if sort_feas_by_obj and len(feasible) > 0:
        f_arr = np.asarray(F)
        if f_arr.ndim > 1:
            f_arr = f_arr[:, 0]
        feasible = feasible[np.argsort(f_arr[feasible])]

    if not return_pop:
        return feasible, infeasible
    else:
        return feasible, infeasible, pop[feasible], pop[infeasible]
