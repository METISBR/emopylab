"""
Standalone implementation of Decision Variables (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike

from core.operator import default_random_state

__all__ = [
    "Variable",
    "BoundedVariable",
    "Real",
    "Integer",
    "Binary",
    "Choice",
    "get",
]


class Variable:
    """Semi-abstract base class for decision variables."""

    def __init__(
        self,
        value: Optional[object] = None,
        active: bool = True,
        flag: str = "default",
    ) -> None:
        super().__init__()
        self.value = value
        self.flag = flag
        self.active = active

    @default_random_state
    def sample(
        self,
        n: Optional[int] = None,
        random_state=None,
    ) -> Union[object, np.ndarray]:
        if n is None:
            return self._sample(1, random_state=random_state)[0]
        else:
            return self._sample(n, random_state=random_state)

    def _sample(self, n: int, random_state=None) -> np.ndarray:
        return np.full(n, self.value)

    def set(self, value: object) -> None:
        self.value = value

    def get(self, **kwargs: Any) -> object:
        return self.value


class BoundedVariable(Variable):
    """Semi-abstract class for bounded decision variables."""

    def __init__(
        self,
        value: Optional[object] = None,
        bounds: Tuple[Optional[object], Optional[object]] = (None, None),
        strict: Optional[Tuple[Optional[object], Optional[object]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value=value, **kwargs)
        self.bounds = bounds
        if strict is None:
            strict = bounds
        self.strict = strict

    @property
    def lb(self) -> object:
        return self.bounds[0]

    @property
    def ub(self) -> object:
        return self.bounds[1]


class Real(BoundedVariable):
    """Real continuous bounded variable."""

    vtype = float

    def _sample(self, n: int, random_state=None) -> np.ndarray:
        low, high = self.bounds
        return random_state.uniform(low=low, high=high, size=n)


class Integer(BoundedVariable):
    """Integer bounded variable."""

    vtype = int

    def _sample(self, n: int, random_state=None) -> np.ndarray:
        low, high = self.bounds
        return random_state.integers(low, high + 1, size=n)


class Binary(BoundedVariable):
    """Binary boolean variable."""

    vtype = bool

    def _sample(self, n: int, random_state=None) -> np.ndarray:
        return random_state.random(size=n) < 0.5


class Choice(Variable):
    """Discrete choice/categorical variable."""

    vtype = object

    def __init__(
        self,
        value: Optional[object] = None,
        options: Optional[ArrayLike] = None,
        all: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value=value, **kwargs)
        self.options = options
        if all is None:
            all = options
        self.all = all

    def _sample(self, n: int, random_state=None) -> np.ndarray:
        return random_state.choice(self.options, size=n)


def get(
    *args: Tuple[Union[Variable, object], ...],
    size: Optional[Union[tuple, int]] = None,
    **kwargs: Any,
) -> Union[tuple, object, None]:
    """Extract values from Variable objects or pass raw values through."""
    if len(args) == 0:
        return None

    ret = []
    for arg in args:
        v = arg.get(**kwargs) if isinstance(arg, Variable) else arg

        if size is not None:
            if isinstance(v, np.ndarray):
                v = np.reshape(v, size)
            else:
                v = np.full(size, v)

        ret.append(v)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
