"""
Standalone implementation of Evaluator and VoidEvaluator (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union
import numpy as np

from core.individual import Individual
from core.population import Population
from core.problem import Problem

__all__ = [
    "Evaluator",
    "VoidEvaluator",
]


class Evaluator:
    """Evaluates populations or individuals on given optimization problems."""

    def __init__(
        self,
        skip_already_evaluated: bool = True,
        evaluate_values_of: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        if evaluate_values_of is None:
            evaluate_values_of = ["F", "G", "H"]
        self.evaluate_values_of = evaluate_values_of
        self.skip_already_evaluated = skip_already_evaluated
        self.callback = callback
        self.n_eval = 0

    def eval(
        self,
        problem: Problem,
        pop: Union[Population, Individual],
        skip_already_evaluated: Optional[bool] = None,
        evaluate_values_of: Optional[List[str]] = None,
        count_evals: bool = True,
        **kwargs: Any,
    ) -> Union[Population, Individual]:
        if evaluate_values_of is None:
            evaluate_values_of = self.evaluate_values_of
        if skip_already_evaluated is None:
            skip_already_evaluated = self.skip_already_evaluated

        is_individual = isinstance(pop, Individual)

        if is_individual:
            pop = Population.create(pop)

        if skip_already_evaluated:
            I = [
                i
                for i, ind in enumerate(pop)
                if not all([e in ind.evaluated for e in evaluate_values_of])
            ]
        else:
            I = list(range(len(pop)))

        if len(I) > 0:
            self._eval(problem, pop[I], evaluate_values_of, **kwargs)

        if count_evals:
            self.n_eval += len(I)

        if self.callback:
            self.callback(pop)

        if is_individual:
            return pop[0]
        else:
            return pop

    def _eval(
        self,
        problem: Problem,
        pop: Population,
        evaluate_values_of: List[str],
        **kwargs: Any,
    ) -> None:
        X = pop.get("X")
        out = problem.evaluate(
            X, return_values_of=evaluate_values_of, return_as_dictionary=True, **kwargs
        )

        for key, val in out.items():
            if val is not None:
                pop.set(key, val)

        pop.apply(lambda ind: ind.evaluated.update(out.keys()))


class VoidEvaluator(Evaluator):
    """Evaluator that assigns placeholder values without calling problem evaluation."""

    def __init__(self, value: float = np.inf, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def eval(self, problem: Problem, pop: Population, **kwargs: Any) -> Population:
        val = self.value
        if val is not None:
            for individual in pop:
                if len(individual.evaluated) == 0:
                    individual.F = np.full(problem.n_obj, val)
                    individual.G = (
                        np.full(problem.n_ieq_constr, val)
                        if problem.n_ieq_constr > 0
                        else None
                    )
                    individual.H = (
                        np.full(problem.n_eq_constr, val)
                        if problem.n_eq_constr > 0
                        else None
                    )
                    individual.CV = np.array([-np.inf])
                    individual.FEAS = np.array([False])
        return pop
