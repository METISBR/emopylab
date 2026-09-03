from __future__ import annotations

"""
FCP constrained benchmark family.

Reference
---------
J. Yuan, H. Liu, Y. Ong, and Z. He.
Indicator-based evolutionary algorithm for solving constrained multi-objective optimization problems.
IEEE Transactions on Evolutionary Computation, 2022, 26(2): 379-391.
"""

import numpy as np
from core.problem import Problem
from util.nds.non_dominated_sorting import NonDominatedSorting


def _uniform_simplex(n: int, m: int = 2) -> np.ndarray:
    n = max(1, int(n))
    if m == 2:
        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([x, 1.0 - x])
    rng = np.random.default_rng(1)
    w = rng.random((n, m))
    return w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-30)


class _BaseFCP(Problem):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=1, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)


class FCP1(_BaseFCP):
    def _calc_pareto_front(self, n_pareto_points=100):
        return 8.5 * _uniform_simplex(n_pareto_points, 2)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        f1 = x[:, 0] * g
        f2 = (1.0 - x[:, 0]) * g

        dis = np.abs(9.0 - g)
        y1 = dis**2 - 0.25
        y2 = (1.2 + np.sin(dis * np.pi)) / (dis + 1e-6)
        c = np.minimum(y1, y2)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = c[:, None]


class FCP2(_BaseFCP):
    def _calc_pareto_front(self, n_pareto_points=200):
        t = np.linspace(0.0, 1.0, int(n_pareto_points) + 1)
        p = np.column_stack([np.cos(0.5 * np.pi * t), np.sin(0.5 * np.pi * t) + 0.2 * np.sin(4.0 * np.pi * t)])
        idx = NonDominatedSorting().do(p, only_non_dominated_front=True)
        return 8.5 * p[idx]

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = 1.0 + 9.0 * np.mean(x[:, 1:] ** 2, axis=1)
        f1 = np.cos(0.5 * np.pi * x[:, 0]) * g
        f2 = (np.sin(0.5 * np.pi * x[:, 0]) + 0.2 * np.sin(4.0 * np.pi * x[:, 0])) * g

        dis = np.abs(9.0 - g)
        y1 = dis**2 - 0.25
        y2 = (1.2 + np.sin(dis * np.pi)) / (dis + 1e-6)
        c = np.minimum(y1, y2)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = c[:, None]


class FCP3(_BaseFCP):
    def _calc_pareto_front(self, n_pareto_points=200):
        t = 0.5 * np.pi * np.linspace(0.0, 1.0, int(n_pareto_points) + 1)
        return 8.5 * np.column_stack([np.cos(t), np.sin(t)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        t = np.mod(np.floor(100.0 * g), 2.0)
        g = g + t * (g - 9.0) ** 2

        f1 = np.cos(0.5 * np.pi * x[:, 0]) * g
        f2 = np.sin(0.5 * np.pi * x[:, 0]) * g

        dis = np.abs(9.0 - g)
        y1 = dis**2 - 0.25
        y2 = (1.2 + np.sin(dis * np.pi)) / (dis + 1e-6)
        c = np.minimum(y1, y2)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = c[:, None]


class FCP4(_BaseFCP):
    def _calc_pareto_front(self, n_pareto_points=200):
        t = np.linspace(0.0, 1.0, int(n_pareto_points) + 1)
        p = np.column_stack([1.0 - t, t + 0.2 * np.sin(4.0 * np.pi * t)])
        idx = NonDominatedSorting().do(p, only_non_dominated_front=True)
        return 8.5 * p[idx]

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        t = np.mod(np.floor(100.0 * g), 2.0)
        g = g + t * (g - 9.0) ** 2

        f1 = (1.0 - x[:, 0]) * g
        f2 = (x[:, 0] + 0.2 * np.sin(4.0 * np.pi * x[:, 0])) * g

        dis = np.abs(9.0 - g)
        y1 = dis**2 - 0.25
        y2 = (1.2 + np.sin(dis * np.pi)) / (dis + 1e-6)
        c = np.minimum(y1, y2)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = c[:, None]


class FCP5(_BaseFCP):
    def _calc_pareto_front(self, n_pareto_points=200):
        t = 0.5 * np.pi * np.linspace(0.0, 1.0, int(n_pareto_points) + 1)
        c1x1 = 0.1 * np.concatenate([9.0 + 0.5 * np.cos(t), 9.0 - 0.5 * np.cos(t)])
        c1g = np.concatenate([3.0 - 0.5 * np.sin(t), 3.0 - 0.5 * np.sin(t)])

        c2x1 = 0.1 * np.concatenate([6.0 + 0.95 * np.cos(t), 6.0 - 0.95 * np.cos(t)])
        c2g = np.concatenate([6.0 - 0.95 * np.sin(t), 6.0 - 0.95 * np.sin(t)])

        c3x1 = 0.1 * np.concatenate([np.sqrt(2.0) + np.sqrt(2.0) * np.cos(t), np.sqrt(2.0) - np.sqrt(2.0) * np.cos(t)])
        c3g = np.concatenate([10.0 - np.sqrt(2.0) * np.sin(t), 10.0 - np.sqrt(2.0) * np.sin(t)])

        x1 = np.concatenate([c1x1, c2x1, c3x1])
        g = np.concatenate([c1g, c2g, c3g])
        p = np.column_stack([x1 * g, (1.0 - x1) * g])
        idx = NonDominatedSorting().do(p, only_non_dominated_front=True)
        return p[idx]

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        f1 = x[:, 0] * g
        f2 = (1.0 - x[:, 0]) * g

        c1 = np.log(np.sqrt((10.0 * x[:, 0] - 9.0) ** 2 + (g - 3.0) ** 2) + 0.5)
        c2 = np.log(np.sqrt((10.0 * x[:, 0] - 6.0) ** 2 + (g - 6.0) ** 2) + 0.05)
        c3 = (10.0 * x[:, 0] - np.sqrt(2.0)) ** 2 + (g - 10.0) ** 2 - 2.0
        c4 = 1.2 + np.sin(np.pi * np.sqrt(c3 + 2.0))
        c = np.min(np.column_stack([c1, c2, c3, c4]), axis=1)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = c[:, None]


for _name in [f"FCP{i}" for i in range(1, 6)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"FCP{i}" for i in range(1, 6)),
    *(f"FCP{i}_JAX" for i in range(1, 6)),
]
