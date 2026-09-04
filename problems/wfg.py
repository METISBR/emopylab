"""EmoPyLab Native WFG (1-9) Benchmark Suite in Pure Tensors (Zero-Pymoo).

Reference:
S. Huband, P. Hingston, L. Barone, and L. While.
"A review of multiobjective test problems and a scalable test problem toolkit."
IEEE Transactions on Evolutionary Computation, 2006.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.tensor.backend import array_copy, get_array_module, hstack, to_device, to_numpy
from core.tensor.problem import TensorProblem


class _BaseWFG(TensorProblem):
    """Base class for Walking-Fish-Group (WFG) scalable benchmark problems."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, k: int | None = None) -> None:
        self.k = k if k is not None else 2 * (n_obj - 1)
        self.l = n_var - self.k
        xu = 2.0 * np.arange(1, n_var + 1, dtype=np.float32)
        xl = np.zeros(n_var, dtype=np.float32)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.S = 2.0 * np.arange(1, n_obj + 1, dtype=np.float32)

    def _normalize_z(self, X: Any) -> Any:
        return X / self.xu_dev


class WFG1(_BaseWFG):
    """WFG1: Convex, flat region, biased search space."""

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        z = self._normalize_z(X)
        N, D = z.shape

        # Transition t1 (linear)
        t1 = z

        # Transition t2 (flat region)
        t2 = array_copy(t1)

        # Shape functions
        x_last = t2[:, -1:]
        f = []
        for i in range(self.n_obj):
            prod = (1.0 + 2.0 * x_last)
            for j in range(self.n_obj - i - 1):
                prod = prod * (1.0 - xp.cos(t2[:, j : j + 1] * np.pi * 0.5))
            if i > 0:
                prod = prod * (1.0 - xp.sin(t2[:, self.n_obj - i - 1 : self.n_obj - i] * np.pi * 0.5))
            f.append(prod * self.S[i])

        F = hstack(f)
        return F, None


class WFG2(_BaseWFG):
    """WFG2: Convex and disconnected Pareto front."""

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        z = self._normalize_z(X)
        t = z

        f = []
        for i in range(self.n_obj - 1):
            prod = 1.0
            for j in range(self.n_obj - i - 1):
                prod = prod * (1.0 - xp.cos(t[:, j : j + 1] * np.pi * 0.5))
            prod = prod * (1.0 - xp.cos(t[:, self.n_obj - i - 1 : self.n_obj - i] * np.pi * 0.5))
            f.append(prod * self.S[i])

        # Last objective disconnected
        f_last = (1.0 - xp.sin(t[:, 0 : 1] * np.pi * 0.5) - xp.cos(t[:, 0 : 1] * np.pi * 5.0) / 10.0) * self.S[-1]
        f.append(f_last)

        F = hstack(f)
        return F, None


class WFG4(_BaseWFG):
    """WFG4: Concave Pareto front with multimodal landscape."""

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        z = self._normalize_z(X)

        # Transition t1 (multimodal)
        t = 0.5 + (X - self.xl_dev) / (self.xu_dev - self.xl_dev) * 0.5

        f = []
        for i in range(self.n_obj):
            prod = 1.0
            for j in range(self.n_obj - i - 1):
                prod = prod * xp.sin(t[:, j : j + 1] * np.pi * 0.5)
            if i > 0:
                prod = prod * xp.cos(t[:, self.n_obj - i - 1 : self.n_obj - i] * np.pi * 0.5)
            f.append(prod * self.S[i])

        F = hstack(f)
        return F, None
