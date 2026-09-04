from __future__ import annotations

"""
MMF benchmark family.

Reference
---------
C. Yue, B. Qu, and J. Liang.
A multi-objective particle swarm optimizer using ring topology for solving multimodal multiobjective problems.
IEEE Transactions on Evolutionary Computation, 2018, 22(5): 805-817.
"""

import numpy as np
from core.problem import Problem


class _BaseMMF(Problem):
    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)


class MMF1(_BaseMMF):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=2, xl=np.array([1.0, -1.0]), xu=np.array([3.0, 1.0]), vtype=float, **kwargs)

    def _calc_pareto_set(self, n_pareto_points=200):
        x1 = np.linspace(1.0, 3.0, int(n_pareto_points))
        x2 = np.sin(6.0 * np.pi * np.abs(x1 - 2.0) + np.pi)
        return np.column_stack([x1, x2])

    def _calc_pareto_front(self, n_pareto_points=200):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = np.abs(x[:, 0] - 2.0)
        f2 = 1.0 - np.sqrt(f1) + 2.0 * (x[:, 1] - np.sin(6.0 * np.pi * f1 + np.pi)) ** 2
        out["F"] = np.column_stack([f1, f2])


class MMF2(_BaseMMF):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=2, xl=np.array([0.0, 0.0]), xu=np.array([1.0, 2.0]), vtype=float, **kwargs)

    def _calc_pareto_set(self, n_pareto_points=200):
        n = max(2, int(n_pareto_points) // 2)
        x1 = np.linspace(0.0, 1.0, n)
        x2 = np.sqrt(x1)
        return np.vstack([np.column_stack([x1, x2]), np.column_stack([x1, x2 + 1.0])])

    def _calc_pareto_front(self, n_pareto_points=200):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        temp = x[:, 1] <= 1.0
        y = np.empty(x.shape[0])
        y[temp] = x[temp, 1] - np.sqrt(x[temp, 0])
        y[~temp] = x[~temp, 1] - 1.0 - np.sqrt(x[~temp, 0])

        f1 = x[:, 0]
        f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 * (4.0 * y**2 - 2.0 * np.cos(20.0 * y * np.pi / np.sqrt(2.0)) + 2.0)
        out["F"] = np.column_stack([f1, f2])


class MMF3(_BaseMMF):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=2, xl=np.array([0.0, 0.0]), xu=np.array([1.0, 1.5]), vtype=float, **kwargs)

    def _calc_pareto_set(self, n_pareto_points=200):
        n = max(2, int(n_pareto_points) // 2)
        x1 = np.linspace(0.0, 1.0, n)
        x2 = np.sqrt(x1)
        return np.vstack([np.column_stack([x1, x2]), np.column_stack([x1, x2 + 0.5])])

    def _calc_pareto_front(self, n_pareto_points=200):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        temp = (x[:, 1] <= 0.5) | ((x[:, 1] < 1.0) & (x[:, 0] > 0.25))
        y = np.empty(x.shape[0])
        y[temp] = x[temp, 1] - np.sqrt(x[temp, 0])
        y[~temp] = x[~temp, 1] - 0.5 - np.sqrt(x[~temp, 0])

        f1 = x[:, 0]
        f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 * (4.0 * y**2 - 2.0 * np.cos(20.0 * y * np.pi / np.sqrt(2.0)) + 2.0)
        out["F"] = np.column_stack([f1, f2])


class MMF4(_BaseMMF):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=2, xl=np.array([-1.0, 0.0]), xu=np.array([1.0, 2.0]), vtype=float, **kwargs)

    def _calc_pareto_set(self, n_pareto_points=200):
        n = max(2, int(n_pareto_points) // 2)
        x1 = np.linspace(-1.0, 1.0, n)
        x2 = np.sin(np.pi * np.abs(x1))
        return np.vstack([np.column_stack([x1, x2]), np.column_stack([x1, x2 + 1.0])])

    def _calc_pareto_front(self, n_pareto_points=200):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([f1, 1.0 - f1**2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        temp = x[:, 1] < 1.0
        y = np.empty(x.shape[0])
        y[temp] = x[temp, 1] - np.sin(np.pi * np.abs(x[temp, 0]))
        y[~temp] = x[~temp, 1] - 1.0 - np.sin(np.pi * np.abs(x[~temp, 0]))

        f1 = np.abs(x[:, 0])
        f2 = 1.0 - x[:, 0] ** 2 + 2.0 * y**2
        out["F"] = np.column_stack([f1, f2])


class MMF5(_BaseMMF):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=2, xl=np.array([1.0, -1.0]), xu=np.array([3.0, 3.0]), vtype=float, **kwargs)

    def _calc_pareto_set(self, n_pareto_points=200):
        n = max(2, int(n_pareto_points) // 2)
        x1 = np.linspace(1.0, 3.0, n)
        x2 = np.sin(6.0 * np.pi * np.abs(x1 - 2.0) + np.pi)
        return np.vstack([np.column_stack([x1, x2]), np.column_stack([x1, x2 + 2.0])])

    def _calc_pareto_front(self, n_pareto_points=200):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        temp = x[:, 1] <= 1.0
        y = np.empty(x.shape[0])
        y[temp] = x[temp, 1] - np.sin(6.0 * np.pi * np.abs(x[temp, 0] - 2.0) + np.pi)
        y[~temp] = x[~temp, 1] - 2.0 - np.sin(6.0 * np.pi * np.abs(x[~temp, 0] - 2.0) + np.pi)

        f1 = np.abs(x[:, 0] - 2.0)
        f2 = 1.0 - np.sqrt(f1) + 2.0 * y**2
        out["F"] = np.column_stack([f1, f2])


class MMF6(_BaseMMF):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=2, xl=np.array([1.0, -1.0]), xu=np.array([3.0, 2.0]), vtype=float, **kwargs)

    def _calc_pareto_set(self, n_pareto_points=200):
        n = max(2, int(n_pareto_points) // 2)
        x1 = np.linspace(1.0, 3.0, n)
        x2 = np.sin(6.0 * np.pi * np.abs(x1 - 2.0) + np.pi)
        return np.vstack([np.column_stack([x1, x2]), np.column_stack([x1, x2 + 1.0])])

    def _calc_pareto_front(self, n_pareto_points=200):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        c = (
            (x[:, 1] <= 0.0)
            | (
                (x[:, 1] <= 1.0)
                & (
                    (x[:, 0] <= 7.0 / 6.0)
                    | ((x[:, 0] > 8.0 / 6.0) & (x[:, 0] <= 9.0 / 6.0))
                    | ((x[:, 0] > 10.0 / 6.0) & (x[:, 0] <= 11.0 / 6.0))
                    | ((x[:, 0] > 13.0 / 6.0) & (x[:, 0] <= 14.0 / 6.0))
                    | ((x[:, 0] > 15.0 / 6.0) & (x[:, 0] <= 16.0 / 6.0))
                    | (x[:, 0] > 17.0 / 6.0)
                )
            )
        )
        y = np.empty(x.shape[0])
        y[c] = x[c, 1] - np.sin(6.0 * np.pi * np.abs(x[c, 0] - 2.0) + np.pi)
        y[~c] = x[~c, 1] - 1.0 - np.sin(6.0 * np.pi * np.abs(x[~c, 0] - 2.0) + np.pi)

        f1 = np.abs(x[:, 0] - 2.0)
        f2 = 1.0 - np.sqrt(f1) + 2.0 * y**2
        out["F"] = np.column_stack([f1, f2])


class MMF7(_BaseMMF):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=2, xl=np.array([1.0, -1.0]), xu=np.array([3.0, 1.0]), vtype=float, **kwargs)

    def _calc_pareto_set(self, n_pareto_points=200):
        x1 = np.linspace(1.0, 3.0, int(n_pareto_points))
        f1 = np.abs(x1 - 2.0)
        x2 = (0.3 * f1**2 * np.cos(24.0 * np.pi * f1 + 4.0 * np.pi) + 0.6 * f1) * np.sin(6.0 * np.pi * f1 + np.pi)
        return np.column_stack([x1, x2])

    def _calc_pareto_front(self, n_pareto_points=200):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = np.abs(x[:, 0] - 2.0)
        center = (0.3 * f1**2 * np.cos(24.0 * np.pi * f1 + 4.0 * np.pi) + 0.6 * f1) * np.sin(6.0 * np.pi * f1 + np.pi)
        f2 = 1.0 - np.sqrt(f1) + (x[:, 1] - center) ** 2
        out["F"] = np.column_stack([f1, f2])


class MMF8(_BaseMMF):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=2, xl=np.array([-np.pi, 0.0]), xu=np.array([np.pi, 9.0]), vtype=float, **kwargs)

    def _calc_pareto_set(self, n_pareto_points=200):
        n = max(2, int(n_pareto_points) // 2)
        x1 = np.linspace(-np.pi, np.pi, n)
        x2 = np.sin(np.abs(x1)) + np.abs(x1)
        return np.vstack([np.column_stack([x1, x2]), np.column_stack([x1, x2 + 4.0])])

    def _calc_pareto_front(self, n_pareto_points=200):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([f1, np.sqrt(1.0 - f1**2)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        temp = x[:, 1] <= 4.0
        y = np.empty(x.shape[0])
        y[temp] = x[temp, 1] - np.sin(np.abs(x[temp, 0])) - np.abs(x[temp, 0])
        y[~temp] = x[~temp, 1] - 4.0 - np.sin(np.abs(x[~temp, 0])) - np.abs(x[~temp, 0])

        f1 = np.sin(np.abs(x[:, 0]))
        f2 = np.sqrt(np.maximum(0.0, 1.0 - f1**2)) + 2.0 * y**2
        out["F"] = np.column_stack([f1, f2])


for _name in [f"MMF{i}" for i in range(1, 9)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"MMF{i}" for i in range(1, 9)),
    *(f"MMF{i}_JAX" for i in range(1, 9)),
]
