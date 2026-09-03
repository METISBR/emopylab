"""
Standalone implementation of Mating (EmoPyLab 2026).
"""

from __future__ import annotations

import math
from typing import Any, Optional

from core.infill import InfillCriterion

__all__ = [
    "Mating",
]


class Mating(InfillCriterion):
    """Combines selection, crossover, and mutation to generate offspring."""

    def __init__(
        self,
        selection: Any = None,
        crossover: Any = None,
        mutation: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(
        self,
        problem,
        pop,
        n_offsprings: int,
        parents: Optional[Any] = None,
        random_state=None,
        **kwargs: Any,
    ):
        n_off_per_crossover = getattr(self.crossover, "n_offsprings", 1) if self.crossover is not None else 1
        n_matings = math.ceil(n_offsprings / n_off_per_crossover)

        if parents is None:
            n_parents = getattr(self.crossover, "n_parents", 2) if self.crossover is not None else 2
            if self.selection is not None:
                parents = self.selection(
                    problem,
                    pop,
                    n_matings,
                    n_parents=n_parents,
                    random_state=random_state,
                    **kwargs,
                )
            else:
                parents = None

        if self.crossover is not None:
            off = self.crossover(
                problem, parents if parents is not None else pop, random_state=random_state, **kwargs
            )
        else:
            off = pop

        if self.mutation is not None:
            off = self.mutation(problem, off, random_state=random_state, **kwargs)

        return off
