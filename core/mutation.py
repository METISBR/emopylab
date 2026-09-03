"""
Standalone implementation of Mutation base operator (EmoPyLab 2026).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional
import numpy as np

from core.operator import Operator, default_random_state
from core.variable import Real, get

__all__ = [
    "Mutation",
]


class Mutation(Operator):
    """Base class for mutation operators."""

    def __init__(
        self,
        prob: float = 1.0,
        prob_var: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prob = Real(prob, bounds=(0.7, 1.0), strict=(0.0, 1.0))
        self.prob_var = (
            Real(prob_var, bounds=(0.0, 0.25), strict=(0.0, 1.0))
            if prob_var is not None
            else None
        )

    @default_random_state
    def do(
        self,
        problem,
        pop,
        inplace: bool = True,
        *args: Any,
        random_state=None,
        **kwargs: Any,
    ):
        if not inplace:
            pop = deepcopy(pop)

        n_mut = len(pop)
        X = pop.get("X")

        Xp = self._do(problem, X, *args, random_state=random_state, **kwargs)

        prob = get(self.prob, size=n_mut)
        mut = random_state.random(size=n_mut) <= prob

        pop[mut].set("X", Xp[mut])
        return pop

    def _do(self, problem, X: np.ndarray, *args: Any, random_state=None, **kwargs: Any) -> np.ndarray:
        return X

    def get_prob_var(self, problem, **kwargs: Any):
        prob_var = (
            self.prob_var
            if self.prob_var is not None
            else (min(0.5, 1.0 / problem.n_var) if problem.n_var > 0 else 0.5)
        )
        return get(prob_var, **kwargs)
