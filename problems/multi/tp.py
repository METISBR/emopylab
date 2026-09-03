from __future__ import annotations

"""
TP robust benchmark family.

Reference
---------
A. Gaspar-Cunha, J. Ferreira, and G. Recio.
Evolutionary robustness analysis for multi-objective optimization: benchmark problems.
Structural and Multidisciplinary Optimization, 2014, 49: 771-793.
"""

import numpy as np
from core.problem import Problem


def _uniform_sphere_2d(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, max(1, int(n)))
    y = 1.0 - x
    f = np.column_stack([x, y])
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-30)


class _BaseTP(Problem):
    def __init__(self, *, n_var: int, xl, xu, delta: float = 0.05, h_disturb: int = 50, n_ieq_constr: int = 0, **kwargs):
        self.delta = float(delta)
        self.H = int(h_disturb)
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=n_ieq_constr, xl=np.asarray(xl, dtype=float), xu=np.asarray(xu, dtype=float), vtype=float, **kwargs)

    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)


class TP1(_BaseTP):
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(n_var=n_var, xl=np.zeros(n_var), xu=np.ones(n_var), **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        y = -((x - 0.6) ** 3 - 0.4**3) / (0.6**3 + 0.4**3)
        return np.column_stack([x, y])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = x[:, 0]
        g = np.mean(x[:, 1:], axis=1)
        h = -((f1 - 0.6) ** 3 - 0.4**3) / (0.6**3 + 0.4**3)
        s = 1.0 / (x[:, 0] + 0.2)
        f2 = h + g * s
        out["F"] = np.column_stack([f1, f2])


class TP2(_BaseTP):
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(n_var=n_var, xl=np.zeros(n_var), xu=np.ones(n_var), **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        return _uniform_sphere_2d(n_pareto_points)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = np.cos(0.5 * np.pi * x[:, 0])
        g = 1.0 + 10.0 * np.mean(x[:, 1:], axis=1)
        f2 = np.sin(0.5 * np.pi * x[:, 0]) * g
        out["F"] = np.column_stack([f1, f2])


class TP3(_BaseTP):
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(n_var=n_var, xl=np.zeros(n_var), xu=np.ones(n_var), **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        f1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        f2 = np.sin(0.5 * np.pi * np.sqrt(1.0 - f1))
        return np.column_stack([f1, f2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = 1.0 - x[:, 0] ** 2
        g = 1.0 + 10.0 * np.mean(x[:, 1:], axis=1)
        f2 = np.sin(0.5 * np.pi * x[:, 0]) * g
        out["F"] = np.column_stack([f1, f2])


class TP4(_BaseTP):
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(n_var=n_var, xl=np.zeros(n_var), xu=np.ones(n_var), **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f1 = (np.exp(x) - 1.0) / (np.exp(1.0) - 1.0)
        f2 = np.sin(4.0 * np.pi * x) / 15.0 - x + 1.0
        return np.column_stack([f1, f2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = (np.exp(x[:, 0]) - 1.0) / (np.exp(1.0) - 1.0)
        g = 1.0 + 10.0 * np.mean(x[:, 1:], axis=1)
        h = np.sin(4.0 * np.pi * x[:, 0]) / 15.0 - x[:, 0] + 1.0
        f2 = h * g
        out["F"] = np.column_stack([f1, f2])


class TP5(_BaseTP):
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(n_var=n_var, xl=np.zeros(n_var), xu=np.ones(n_var), **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        y = np.sin(4.0 * np.pi * x) / 15.0 - x + 1.0
        return np.column_stack([x, y])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = x[:, 0]
        g = 1.0 + 10.0 * np.mean(x[:, 1:], axis=1)
        h = np.sin(4.0 * np.pi * x[:, 0]) / 15.0 - x[:, 0] + 1.0
        f2 = h * g
        out["F"] = np.column_stack([f1, f2])


class TP6(_BaseTP):
    def __init__(self, n_var: int = 5, **kwargs):
        xl = np.concatenate(([0.0], -np.ones(n_var - 1)))
        xu = np.concatenate(([1.0], np.ones(n_var - 1)))
        super().__init__(n_var=n_var, xl=xl, xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - x**2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = x[:, 0]
        h = 1.0 - x[:, 0] ** 2
        g = np.sum(10.0 - 10.0 * np.cos(4.0 * np.pi * x[:, 1:]) + x[:, 1:] ** 2, axis=1)
        s = 1.0 / (0.2 + x[:, 0]) + x[:, 0] ** 2
        f2 = h + g * s
        out["F"] = np.column_stack([f1, f2])


class TP7(_BaseTP):
    def __init__(self, n_var: int = 5, **kwargs):
        xl = np.concatenate(([0.0], -np.ones(n_var - 1)))
        xu = np.concatenate(([1.0], np.ones(n_var - 1)))
        super().__init__(n_var=n_var, xl=xl, xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - x**2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = x[:, 0]
        h = 1.0 - x[:, 0] ** 2
        g = np.sum(10.0 - 10.0 * np.cos(4.0 * np.pi * x[:, 1:]) + x[:, 1:] ** 2, axis=1)
        s = 1.0 / (0.2 + x[:, 0]) + 10.0 * x[:, 0] ** 2
        f2 = h + g * s
        out["F"] = np.column_stack([f1, f2])


class TP8(_BaseTP):
    def __init__(self, n_var: int = 5, **kwargs):
        xl = np.concatenate(([0.0, 0.0], -np.ones(n_var - 2)))
        xu = np.concatenate(([1.0, 1.0], np.ones(n_var - 2)))
        super().__init__(n_var=n_var, xl=xl, xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = x[:, 0]
        h = 2.0 - 0.8 * np.exp(-((x[:, 1] - 0.35) / 0.25) ** 2) - np.exp(-((x[:, 1] - 0.85) / 0.03) ** 2)
        g = 50.0 * np.sum(x[:, 2:] ** 2, axis=1)
        s = 1.0 - np.sqrt(f1)
        f2 = h * (g + s)
        out["F"] = np.column_stack([f1, f2])


class TP9(_BaseTP):
    def __init__(self, n_var: int = 5, **kwargs):
        xl = np.concatenate(([0.0, 0.0], -np.ones(n_var - 2)))
        xu = np.concatenate(([1.0, 1.0], np.ones(n_var - 2)))
        super().__init__(n_var=n_var, xl=xl, xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        y = (1.0 - np.sqrt(x)) * (1.0 - x - 0.8 * np.exp(-4.0))
        return np.column_stack([x, y])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = x[:, 0]
        h = 2.0 - f1 - 0.8 * np.exp(-((f1 + x[:, 1] - 0.35) / 0.25) ** 2) - np.exp(-((f1 + x[:, 1] - 0.85) / 0.03) ** 2)
        g = 50.0 * np.sum(x[:, 2:] ** 2, axis=1)
        s = 1.0 - np.sqrt(f1)
        f2 = h * (g + s)
        out["F"] = np.column_stack([f1, f2])


class TP10(_BaseTP):
    def __init__(self, **kwargs):
        super().__init__(n_var=3, xl=np.array([0.0, 0.0, 1.0]), xu=np.array([10.0, 10.0, 3.0]), n_ieq_constr=2, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=1):
        _ = n_pareto_points
        return np.array([[100.0, 100.0]])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        f1 = x[:, 0] * np.sqrt(16.0 + x[:, 2] ** 2) + x[:, 1] * np.sqrt(1.0 + x[:, 2] ** 2)
        f2 = 20.0 * np.sqrt(16.0 + x[:, 2] ** 2) / np.maximum(x[:, 0] * x[:, 2], 1e-30)

        c1 = 20.0 * np.sqrt(16.0 + x[:, 2] ** 2) - 100.0 * x[:, 0] * x[:, 2]
        c2 = 80.0 * np.sqrt(1.0 + x[:, 2] ** 2) - 100.0 * x[:, 1] * x[:, 2]

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2])


for _name in [f"TP{i}" for i in range(1, 11)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"TP{i}" for i in range(1, 11)),
    *(f"TP{i}_JAX" for i in range(1, 11)),
]
