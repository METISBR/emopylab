"""EmoPyLab Native Vectorized Problem Contract.

Defines the base TensorProblem class where evaluations f: R^{N x D} -> R^{N x M}
and constraints g: R^{N x D} -> R^{N x K} execute as pure tensor transforms in batch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np

from core.tensor.backend import clip_bounds, get_array_module, to_device, to_numpy


class TensorProblem(ABC):
    """Abstract Base Class for Tensor-Native Evolutionary Problems (Zero-Pymoo)."""

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        n_ieq_constr: int = 0,
        n_eq_constr: int = 0,
        xl: Optional[Sequence[float] | np.ndarray] = None,
        xu: Optional[Sequence[float] | np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.n_ieq_constr = int(n_ieq_constr)
        self.n_eq_constr = int(n_eq_constr)
        self.n_constr = self.n_ieq_constr + self.n_eq_constr
        self.name = str(name or self.__class__.__name__)

        # Lower and upper decision bounds
        if xl is None:
            xl = np.zeros(self.n_var, dtype=np.float32)
        elif np.isscalar(xl):
            xl = np.full(self.n_var, float(xl), dtype=np.float32)
        else:
            xl = np.asarray(xl, dtype=np.float32).reshape(-1)

        if xu is None:
            xu = np.ones(self.n_var, dtype=np.float32)
        elif np.isscalar(xu):
            xu = np.full(self.n_var, float(xu), dtype=np.float32)
        else:
            xu = np.asarray(xu, dtype=np.float32).reshape(-1)

        if xl.size != self.n_var or xu.size != self.n_var:
            raise ValueError(f"Bounds dimension mismatch: xl={xl.size}, xu={xu.size}, expected n_var={self.n_var}")
        self.xl = xl
        self.xu = xu
        self.xl_cpu = xl
        self.xu_cpu = xu
        self.xl_dev = to_device(xl)
        self.xu_dev = to_device(xu)

    def has_bounds(self) -> bool:
        """Returns True if the problem has defined decision variable bounds."""
        return self.xl is not None and self.xu is not None

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns tuple of (lower_bounds, upper_bounds) as numpy arrays."""
        return self.xl, self.xu

    def has_constraints(self) -> bool:
        """Returns True if the problem has inequality or equality constraints."""
        return self.n_constr > 0

    def clamp(self, X: Any) -> Any:
        """Clamps decision matrix X to [xl, xu] on active device."""
        return clip_bounds(X, self.xl_dev, self.xu_dev)

    @abstractmethod
    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        """Pure tensor evaluation function.

        Args:
            X: Decision matrix tensor [N, D].

        Returns:
            Tuple (F, G):
                F: Objectives tensor [N, M].
                G: Constraints tensor [N, K] (or None if unconstrained).
        """
        raise NotImplementedError

    def evaluate(self, X: Any, *args: Any, return_values_of: Any = None, **kwargs: Any) -> Any:
        """Evaluates batch of solutions X with automatic bound enforcement."""
        if isinstance(X, np.ndarray) or not hasattr(X, "shape"):
            # Handle population objects or standard arrays
            pass
        X_clamped = self.clamp(X)
        F, G = self._evaluate(X_clamped)
        if return_values_of is not None:
            res_dict = {}
            if "F" in return_values_of:
                res_dict["F"] = to_numpy(F)
            if "G" in return_values_of:
                res_dict["G"] = to_numpy(G) if G is not None else None
            return res_dict
        return F, G

    def pareto_front(self, n_points: int = 100) -> Optional[np.ndarray]:
        """Generates or returns true analytical Pareto front (if available)."""
        return None
