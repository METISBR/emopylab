"""EmoPyLab Native DTLZ Benchmark Suite in Pure Tensors (Zero-Pymoo)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.tensor.backend import get_array_module, hstack, to_device, to_numpy
from core.tensor.problem import TensorProblem


class DTLZ1(TensorProblem):
    """Deb-Thiele-Laumanns-Zitzler test problem 1 (Linear Hyperplane)."""

    def __init__(self, n_var: int = 7, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.k = self.n_var - self.n_obj + 1

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        X_M = X[:, self.n_obj - 1 :]
        g = 100.0 * (self.k + xp.sum((X_M - 0.5) ** 2 - xp.cos(20.0 * np.pi * (X_M - 0.5)), axis=1, keepdims=True))

        f = []
        for i in range(self.n_obj):
            prod = 0.5 * (1.0 + g)
            for j in range(self.n_obj - i - 1):
                prod = prod * X[:, j : j + 1]
            if i > 0:
                prod = prod * (1.0 - X[:, self.n_obj - i - 1 : self.n_obj - i])
            f.append(prod)

        F = hstack(f)
        return F, None


class DTLZ2(TensorProblem):
    """Deb-Thiele-Laumanns-Zitzler test problem 2 (Spherical Manifold)."""

    def __init__(self, n_var: int = 12, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        X_M = X[:, self.n_obj - 1 :]
        g = xp.sum((X_M - 0.5) ** 2, axis=1, keepdims=True)

        f = []
        for i in range(self.n_obj):
            val = 1.0 + g
            for j in range(self.n_obj - i - 1):
                val = val * xp.cos(X[:, j : j + 1] * np.pi * 0.5)
            if i > 0:
                val = val * xp.sin(X[:, self.n_obj - i - 1 : self.n_obj - i] * np.pi * 0.5)
            f.append(val)

        F = hstack(f)
        return F, None
