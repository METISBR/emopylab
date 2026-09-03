"""EmoPyLab Native Constrained and Real-World Benchmark Problems (Zero-Pymoo).

Includes C-DTLZ, Welded Beam Design, and Pressure Vessel Design.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.tensor.backend import get_array_module, hstack, to_device, to_numpy
from core.tensor.problem import TensorProblem


class CDTLZ2(TensorProblem):
    """Constrained DTLZ2 benchmark with non-linear spherical boundary constraint."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, r: float = 0.5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, xl=0.0, xu=1.0)
        self.r = float(r)

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

        # Constraint: (sum(f_m^2) - r^2) <= 0
        f_norm_sq = xp.sum(F ** 2, axis=1, keepdims=True)
        G = -(f_norm_sq - (self.r ** 2))

        return F, G


class WeldedBeam(TensorProblem):
    """Welded Beam Design Problem (2 Objectives, 4 Variables, 4 Constraints)."""

    def __init__(self) -> None:
        xl = np.array([0.125, 0.1, 0.1, 0.125], dtype=np.float32)
        xu = np.array([5.0, 10.0, 10.0, 5.0], dtype=np.float32)
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=4, xl=xl, xu=xu)

    def _evaluate(self, X: Any) -> tuple[Any, Optional[Any]]:
        xp = get_array_module()
        x1 = X[:, 0:1]  # h
        x2 = X[:, 1:2]  # l
        x3 = X[:, 2:3]  # t
        x4 = X[:, 3:4]  # b
        f1 = 1.10471 * (x1 ** 2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
        f2 = 2.1952 / ((x3 ** 3) * x4)

        F = hstack([f1, f2])

        # Constraints (shear stress, bending stress, geometry, buckling)
        g1 = 13600.0 - (6000.0 / (np.sqrt(2.0) * x1 * x2))
        g2 = 30000.0 - (504000.0 / (x4 * (x3 ** 2)))
        g3 = x4 - x1
        g4 = 6000.0 - (64746.022 * (1.0 - 0.0282346 * x3) * x3 * (x4 ** 3))

        G = hstack([-g1, -g2, -g3, -g4])
        return F, G
