"""
Standalone implementation of Selection base operator (EmoPyLab 2026).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional
import numpy as np

from core.operator import Operator, default_random_state

__all__ = [
    "Selection",
]


class Selection(Operator):
    """Base class for parent selection operators."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @default_random_state
    def do(
        self,
        problem,
        pop,
        n_select: int,
        n_parents: int = 2,
        to_pop: bool = True,
        *args: Any,
        random_state=None,
        **kwargs: Any,
    ):
        ret = self._do(
            problem, pop, n_select, n_parents, *args, random_state=random_state, **kwargs
        )

        if to_pop and isinstance(ret, np.ndarray) and np.issubdtype(ret.dtype, np.integer):
            ret = pop[ret]

        return ret

    @abstractmethod
    def _do(
        self,
        problem,
        pop,
        n_select: int,
        n_parents: int,
        *args: Any,
        random_state=None,
        **kwargs: Any,
    ):
        pass
