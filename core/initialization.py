"""
Standalone implementation of Initialization (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from core.duplicate import DuplicateElimination, NoDuplicateElimination
from core.operator import Operator, default_random_state
from core.population import Population
from core.problem import Problem, at_least_2d_array
from core.repair import NoRepair, Repair

__all__ = [
    "Initialization",
]


class Initialization:
    """Initial population sampler with repair and duplicate filtering."""

    def __init__(
        self,
        sampling: Any,
        repair: Optional[Repair] = None,
        eliminate_duplicates: Optional[DuplicateElimination] = None,
    ) -> None:
        super().__init__()
        self.sampling = sampling
        self.eliminate_duplicates = (
            eliminate_duplicates
            if eliminate_duplicates is not None
            else NoDuplicateElimination()
        )
        self.repair = repair if repair is not None else NoRepair()

    @default_random_state
    def do(
        self,
        problem: Problem,
        n_samples: int,
        random_state=None,
        **kwargs: Any,
    ) -> Population:
        if isinstance(self.sampling, Population):
            pop = self.sampling
        else:
            if isinstance(self.sampling, np.ndarray):
                sampling_arr = at_least_2d_array(self.sampling)
                pop = Population.new(X=sampling_arr)
            else:
                pop = self.sampling(
                    problem, n_samples, random_state=random_state, **kwargs
                )

        not_eval_yet = [k for k in range(len(pop)) if len(pop[k].evaluated) == 0]
        if len(not_eval_yet) > 0:
            pop[not_eval_yet] = self.repair(
                problem, pop[not_eval_yet], random_state=random_state, **kwargs
            )

        pop = self.eliminate_duplicates.do(pop)
        return pop
