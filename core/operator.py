"""
Standalone implementation of base Operator (EmoPyLab 2026).
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Optional
import numpy as np


def default_random_state(func_or_seed=None, *, seed=None):
    """
    Decorator that provides a default random state to functions/methods.
    If random_state is provided, it takes precedence.
    """
    def decorator(func, default_seed=None):
        def wrapper(*args, random_state=None, **kwargs):
            if random_state is None:
                seed_to_use = kwargs.pop("seed", default_seed)
                random_state = np.random.default_rng(seed_to_use)
            return func(*args, random_state=random_state, **kwargs)
        return wrapper

    if func_or_seed is None:
        return lambda func: decorator(func, seed)
    elif callable(func_or_seed):
        return decorator(func_or_seed, None)
    else:
        return lambda func: decorator(func, func_or_seed)


class Operator(abc.ABC):
    """Base class for all evolutionary operators."""

    def __init__(
        self,
        name: Optional[str] = None,
        vtype: Optional[type] = None,
        repair: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.vtype = vtype
        self.repair = repair

    @default_random_state
    def do(self, problem, elem, *args, random_state=None, **kwargs):
        return self._do(problem, elem, *args, random_state=random_state, **kwargs)

    @abc.abstractmethod
    def _do(self, problem, elem, *args, random_state=None, **kwargs):
        pass

    def __call__(self, problem, elem, *args, to_numpy: bool = False, **kwargs):
        out = self.do(problem, elem, *args, **kwargs)

        if self.vtype is not None:
            for ind in out:
                ind.X = np.asarray(ind.X).astype(self.vtype)

        if self.repair is not None:
            self.repair.do(problem, out)

        if to_numpy:
            out = np.array([ind.X for ind in out])

        return out
