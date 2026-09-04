"""
Standalone implementation of Sampling base operator (EmoPyLab 2026).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional
import numpy as np

from core.operator import Operator, default_random_state
from core.population import Population

__all__ = [
    "Sampling",
]


class Sampling(Operator):
    """Base class for population sampling strategies."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @default_random_state
    def do(self, problem, n_samples: int, *args: Any, random_state=None, **kwargs: Any):
        val = self._do(problem, n_samples, *args, random_state=random_state, **kwargs)
        if isinstance(val, Population):
            return val
        elif isinstance(val, np.ndarray):
            if val.dtype == object and len(val) > 0 and hasattr(val[0], "X"):
                return Population(list(val))
            return Population.new("X", val)
        elif isinstance(val, (list, tuple)):
            if len(val) > 0 and hasattr(val[0], "X"):
                return Population(list(val))
            return Population.new("X", np.asarray(val))
        return val

    @abstractmethod
    def _do(self, problem, n_samples: int, *args: Any, random_state=None, **kwargs: Any) -> np.ndarray:
        pass
