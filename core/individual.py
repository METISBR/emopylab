"""
Standalone implementation of Individual and constraint violation utilities (EmoPyLab 2026).
"""

from __future__ import annotations

import copy
from typing import Any, Optional, Tuple, Union
from warnings import warn
import numpy as np

__all__ = [
    "default_config",
    "Individual",
    "calc_cv",
    "constr_to_cv",
]


def default_config() -> dict:
    """Get default constraint violation configuration settings."""
    return dict(
        cache=True,
        cv_eps=0.0,
        cv_ieq=dict(scale=None, eps=0.0, pow=None, func=np.sum),
        cv_eq=dict(scale=None, eps=1e-4, pow=None, func=np.sum),
    )


class Individual:
    """Base class representing an individual in population-based optimization."""

    default_config = default_config

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self._X = None
        self._F = None
        self._G = None
        self._H = None
        self._dF = None
        self._dG = None
        self._dH = None
        self._ddF = None
        self._ddG = None
        self._ddH = None
        self._CV = None
        self.evaluated = None

        self.reset()
        self.data = {}

        if config is None:
            config = Individual.default_config()
        self.config = config

        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            elif "_" + k in self.__dict__:
                self.__dict__["_" + k] = v
            else:
                self.data[k] = v

    def reset(self, data: bool = True) -> None:
        empty = np.array([])
        self._X = empty
        self._F = empty
        self._G = empty
        self._H = empty
        self._dF = empty
        self._dG = empty
        self._dH = empty
        self._ddF = empty
        self._ddG = empty
        self._ddH = empty
        self._CV = None
        if data:
            self.data = {}
        self.evaluated = set()

    def has(self, key: str) -> bool:
        return hasattr(self.__class__, key) or key in self.data or hasattr(self, key)

    # -------------------------------------------------------
    # Values
    # -------------------------------------------------------

    @property
    def X(self) -> np.ndarray:
        if isinstance(self._X, np.ndarray) and self._X.ndim > 1:
            return self._X.squeeze()
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                value = np.squeeze(value)
            if value.ndim == 0:
                value = np.array([value])
        self._X = value

    @property
    def F(self) -> np.ndarray:
        if isinstance(self._F, np.ndarray) and self._F.ndim > 1:
            return self._F.squeeze()
        return self._F

    @F.setter
    def F(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                value = np.squeeze(value)
            if value.ndim == 0:
                value = np.array([value])
        self._F = value

    @property
    def G(self) -> np.ndarray:
        if isinstance(self._G, np.ndarray) and self._G.ndim > 1:
            return self._G.squeeze()
        return self._G

    @G.setter
    def G(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                value = np.squeeze(value)
            if value.ndim == 0:
                value = np.array([value])
        self._G = value

    @property
    def H(self) -> np.ndarray:
        if isinstance(self._H, np.ndarray) and self._H.ndim > 1:
            return self._H.squeeze()
        return self._H

    @H.setter
    def H(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                value = np.squeeze(value)
            if value.ndim == 0:
                value = np.array([value])
        self._H = value
    @property
    def CV(self) -> np.ndarray:
        config = self.config
        cache = config.get("cache", True)

        if cache and self._CV is not None:
            return self._CV
        else:
            self._CV = np.array([calc_cv(G=self.G, H=self.H, config=config)])
            return self._CV

    @CV.setter
    def CV(self, value: np.ndarray) -> None:
        self._CV = value

    @property
    def FEAS(self) -> np.ndarray:
        eps = self.config.get("cv_eps", 0.0)
        return self.CV <= eps

    # -------------------------------------------------------
    # Gradients & Hessians
    # -------------------------------------------------------

    @property
    def dF(self) -> np.ndarray:
        return self._dF

    @dF.setter
    def dF(self, value: np.ndarray) -> None:
        self._dF = value

    @property
    def dG(self) -> np.ndarray:
        return self._dG

    @dG.setter
    def dG(self, value: np.ndarray) -> None:
        self._dG = value

    @property
    def dH(self) -> np.ndarray:
        return self._dH

    @dH.setter
    def dH(self, value: np.ndarray) -> None:
        self._dH = value

    @property
    def ddF(self) -> np.ndarray:
        return self._ddF

    @ddF.setter
    def ddF(self, value: np.ndarray) -> None:
        self._ddF = value

    @property
    def ddG(self) -> np.ndarray:
        return self._ddG

    @ddG.setter
    def ddG(self, value: np.ndarray) -> None:
        self._ddG = value

    @property
    def ddH(self) -> np.ndarray:
        return self._ddH

    @ddH.setter
    def ddH(self, value: np.ndarray) -> None:
        self._ddH = value

    # -------------------------------------------------------
    # Convenience
    # -------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self.X

    @property
    def f(self) -> float:
        return self.F[0] if self.F is not None and len(self.F) > 0 else None

    @property
    def cv(self) -> Union[float, None]:
        if self.CV is None:
            return None
        return self.CV[0]

    @property
    def feas(self) -> bool:
        return bool(self.FEAS[0]) if self.FEAS is not None and len(self.FEAS) > 0 else True

    @property
    def feasible(self) -> np.ndarray:
        warn(
            "The ``feasible`` property for ``Individual`` is deprecated, use ``FEAS`` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.FEAS

    # -------------------------------------------------------
    # Methods
    # -------------------------------------------------------

    def set_by_dict(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, key: str, value: object) -> Individual:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.data[key] = value
        return self

    def get(self, *keys: str) -> Union[tuple, object]:
        ret = []
        for key in keys:
            if key in ("feasible", "FEAS"):
                v = self.FEAS
            elif hasattr(self, key):
                v = getattr(self, key)
            elif hasattr(self, "_" + key):
                v = getattr(self, "_" + key)
            elif key in self.data:
                v = self.data[key]
            else:
                v = None
            ret.append(v)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def duplicate(self, key: str, new_key: str) -> None:
        self.set(new_key, self.get(key))

    def new(self) -> Individual:
        return self.__class__()

    def copy(self, other: Optional[Individual] = None, deep: bool = True) -> Individual:
        obj = self.new()
        src = self if other is None else other
        if deep:
            obj.__dict__ = copy.deepcopy(src.__dict__)
        else:
            obj.__dict__ = copy.copy(src.__dict__)
        return obj

def calc_cv(
    G: Optional[np.ndarray] = None,
    H: Optional[np.ndarray] = None,
    config: Optional[dict] = None,
) -> np.ndarray:
    """Calculate constraint violation."""
    if G is None:
        G = np.array([])
    if H is None:
        H = np.array([])
    if config is None:
        config = Individual.default_config()

    if G is None or len(G) == 0:
        ieq_cv = [0.0]
    elif G.ndim == 1:
        ieq_cv = [constr_to_cv(G, **config.get("cv_ieq", {}))]
    else:
        ieq_cv = [constr_to_cv(g, **config.get("cv_ieq", {})) for g in G]

    if H is None or len(H) == 0:
        eq_cv = [0.0]
    elif H.ndim == 1:
        eq_cv = [constr_to_cv(np.abs(H), **config.get("cv_eq", {}))]
    else:
        eq_cv = [constr_to_cv(np.abs(h), **config.get("cv_eq", {})) for h in H]

    return np.array(ieq_cv) + np.array(eq_cv)


def constr_to_cv(
    c: Union[np.ndarray, None],
    eps: float = 0.0,
    scale: Optional[float] = None,
    pow: Optional[float] = None,
    func: object = np.mean,
) -> float:
    """Convert raw constraint vector to aggregated violation float."""
    if c is None or len(c) == 0:
        return 0.0

    c = np.maximum(0.0, c - eps)
    if scale is not None:
        c = c / scale
    if pow is not None:
        c = c ** pow

    return float(func(c))
