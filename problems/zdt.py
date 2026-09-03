"""EmoPyLab Native ZDT Benchmark Suite in Pure Tensors (Zero-Pymoo)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.tensor.backend import get_array_module, hstack, to_device, to_numpy
from core.tensor.problem import TensorProblem


class ZDT1(TensorProblem):
    """Zitzler-Deb-Thiele test problem 1 (Convex Pareto Front)."""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        f1 = X[:, 0:1]
        g = 1.0 + 9.0 * xp.mean(X[:, 1:], axis=1, keepdims=True)
        f2 = g * (1.0 - xp.sqrt(f1 / g))
        F = hstack([f1, f2])
        return F, None

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        x = np.linspace(0.0, 1.0, int(max(2, n_points)), dtype=np.float32)
        return np.column_stack([x, 1.0 - np.sqrt(x)])


class ZDT2(TensorProblem):
    """Zitzler-Deb-Thiele test problem 2 (Concave Pareto Front)."""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        f1 = X[:, 0:1]
        g = 1.0 + 9.0 * xp.mean(X[:, 1:], axis=1, keepdims=True)
        f2 = g * (1.0 - (f1 / g) ** 2)
        F = hstack([f1, f2])
        return F, None

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        x = np.linspace(0.0, 1.0, int(max(2, n_points)), dtype=np.float32)
        return np.column_stack([x, 1.0 - x ** 2])


class ZDT3(TensorProblem):
    """Zitzler-Deb-Thiele test problem 3 (Disconnected Pareto Front)."""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        f1 = X[:, 0:1]
        g = 1.0 + 9.0 * xp.mean(X[:, 1:], axis=1, keepdims=True)
        f2 = g * (1.0 - xp.sqrt(f1 / g) - (f1 / g) * xp.sin(10.0 * np.pi * f1))
        F = hstack([f1, f2])
        return F, None
