"""
Standalone implementation of InfillCriterion (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.duplicate import DuplicateElimination, NoDuplicateElimination
from core.population import Population
from core.repair import NoRepair, Repair

__all__ = [
    "InfillCriterion",
]


class InfillCriterion:
    """Base class for generating infill / offspring populations."""

    def __init__(
        self,
        repair: Optional[Repair] = None,
        eliminate_duplicates: Optional[DuplicateElimination] = None,
        n_max_iterations: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_max_iterations = n_max_iterations
        self.eliminate_duplicates = (
            eliminate_duplicates
            if eliminate_duplicates is not None
            else NoDuplicateElimination()
        )
        self.repair = repair if repair is not None else NoRepair()

    def __call__(self, problem, pop, n_offsprings: int, random_state=None, **kwargs: Any):
        return self.do(problem, pop, n_offsprings, random_state=random_state, **kwargs)

    def do(
        self,
        problem,
        pop,
        n_offsprings: int,
        random_state=None,
        n_max_iterations: Optional[int] = None,
        **kwargs: Any,
    ):
        if n_max_iterations is None:
            n_max_iterations = self.n_max_iterations

        off = Population.create()
        n_infills = 0

        while len(off) < n_offsprings:
            n_remaining = n_offsprings - len(off)

            _off = self._do(
                problem, pop, n_remaining, random_state=random_state, **kwargs
            )

            _off = self.repair(problem, _off, random_state=random_state, **kwargs)
            _off = self.eliminate_duplicates.do(_off, pop, off)

            if len(off) + len(_off) > n_offsprings:
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]

            off = Population.merge(off, _off)
            n_infills += 1

            if n_infills >= n_max_iterations:
                break

        return off

    def _do(self, problem, pop, n_offsprings: int, random_state=None, **kwargs: Any):
        return Population.create()
