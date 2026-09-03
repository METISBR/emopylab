from __future__ import annotations

"""
BT benchmark family.

Reference
---------
H. Li, Q. Zhang, and J. Deng.
Biased multiobjective optimization and decomposition algorithm.
IEEE Transactions on Cybernetics, 2017, 47(1): 52-66.
"""

import numpy as np
from core.problem import Problem
from util.nds.non_dominated_sorting import NonDominatedSorting


class _BaseBT(Problem):
    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)


def _bt_bias(y, eps):
    return y**2 + (1.0 - np.exp(-(y**2) / eps)) / 5.0


class BT1(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(1, d, 2)
        i2 = np.arange(2, d, 2)

        idx = np.arange(1, d + 1, dtype=float)
        y = x - np.sin(np.tile(idx, (n, 1)) * np.pi / (2.0 * d))

        f1 = x[:, 0] + np.sum(_bt_bias(y[:, i1], 1e-10), axis=1)
        f2 = 1.0 - np.sqrt(x[:, 0]) + np.sum(_bt_bias(y[:, i2], 1e-10), axis=1)
        out["F"] = np.column_stack([f1, f2])


class BT2(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(1, d, 2)
        i2 = np.arange(2, d, 2)

        idx = np.arange(1, d + 1, dtype=float)
        y = x - np.sin(np.tile(idx, (n, 1)) * np.pi / (2.0 * d))

        f1 = x[:, 0] + np.sum(y[:, i1] ** 2 + np.abs(y[:, i1]) ** (1.0 / 5.0) / 5.0, axis=1)
        f2 = 1.0 - np.sqrt(x[:, 0]) + np.sum(y[:, i2] ** 2 + np.abs(y[:, i2]) ** (1.0 / 5.0) / 5.0, axis=1)
        out["F"] = np.column_stack([f1, f2])


class BT3(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(1, d, 2)
        i2 = np.arange(2, d, 2)

        idx = np.arange(1, d + 1, dtype=float)
        y = x - np.sin(np.tile(idx, (n, 1)) * np.pi / (2.0 * d))
        x0 = np.abs(x[:, 0]) ** 0.02

        f1 = x0 + np.sum(_bt_bias(y[:, i1], 1e-8), axis=1)
        f2 = 1.0 - np.sqrt(x0) + np.sum(_bt_bias(y[:, i2], 1e-8), axis=1)
        out["F"] = np.column_stack([f1, f2])


class BT4(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(1, d, 2)
        i2 = np.arange(2, d, 2)

        idx = np.arange(1, d + 1, dtype=float)
        y = x - np.sin(np.tile(idx, (n, 1)) * np.pi / (2.0 * d))

        x0 = x[:, 0]
        t1 = x0 < 0.25
        t2 = (x0 >= 0.25) & (x0 < 0.5)
        t3 = (x0 >= 0.5) & (x0 < 0.75)
        t4 = ~(t1 | t2 | t3)
        x0b = np.empty_like(x0)
        x0b[t1] = (1.0 - (1.0 - 4.0 * x0[t1]) ** 0.06) / 4.0
        x0b[t2] = (1.0 + (4.0 * x0[t2] - 1.0) ** 0.06) / 4.0
        x0b[t3] = (3.0 - (3.0 - 4.0 * x0[t3]) ** 0.06) / 4.0
        x0b[t4] = (3.0 + (4.0 * x0[t4] - 3.0) ** 0.06) / 4.0

        f1 = x0b + np.sum(_bt_bias(y[:, i1], 1e-8), axis=1)
        f2 = 1.0 - np.sqrt(x0b) + np.sum(_bt_bias(y[:, i2], 1e-8), axis=1)
        out["F"] = np.column_stack([f1, f2])


class BT5(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=500):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f2 = (1.0 - x) * (1.0 - x * np.sin(8.5 * np.pi * x))
        pf = np.column_stack([x, f2])
        rank = NonDominatedSorting().do(pf, only_non_dominated_front=True)
        return pf[rank]

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(1, d, 2)
        i2 = np.arange(2, d, 2)

        idx = np.arange(1, d + 1, dtype=float)
        y = x - np.sin(np.tile(idx, (n, 1)) * np.pi / (2.0 * d))

        f1 = x[:, 0] + np.sum(_bt_bias(y[:, i1], 1e-10), axis=1)
        f2 = (1.0 - x[:, 0]) * (1.0 - x[:, 0] * np.sin(8.5 * np.pi * x[:, 0])) + np.sum(_bt_bias(y[:, i2], 1e-10), axis=1)
        out["F"] = np.column_stack([f1, f2])


class BT6(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(1, d, 2)
        i2 = np.arange(2, d, 2)

        p = 0.5 + 1.5 * (np.arange(d, dtype=float) / (d - 1.0))
        y = x - x[:, [0]] ** p[None, :]

        f1 = x[:, 0] + np.sum(_bt_bias(y[:, i1], 1e-4), axis=1)
        f2 = 1.0 - np.sqrt(x[:, 0]) + np.sum(_bt_bias(y[:, i2], 1e-4), axis=1)
        out["F"] = np.column_stack([f1, f2])


class BT7(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(n - 1)))
        xu = np.ones(n)
        super().__init__(n_var=n, n_obj=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(1, d, 2)
        i2 = np.arange(2, d, 2)

        y = x - np.sin(6.0 * np.pi * x[:, [0]])

        f1 = x[:, 0] + np.sum(_bt_bias(y[:, i1], 1e-3), axis=1)
        f2 = 1.0 - np.sqrt(x[:, 0]) + np.sum(_bt_bias(y[:, i2], 1e-3), axis=1)
        out["F"] = np.column_stack([f1, f2])


class BT8(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(1, d, 2)
        i2 = np.arange(2, d, 2)

        p = 0.5 + 1.5 * (np.arange(d, dtype=float) / (d - 1.0))
        y = x - x[:, [0]] ** p[None, :]
        dy = y**2 + (1.0 - np.exp(-(y**2) / 1e-3)) / 5.0

        f1 = x[:, 0] + np.sum(4.0 * dy[:, i1] ** 2 - np.cos(8.0 * np.pi * dy[:, i1]) + 1.0, axis=1)
        f2 = 1.0 - np.sqrt(x[:, 0]) + np.sum(4.0 * dy[:, i2] ** 2 - np.cos(8.0 * np.pi * dy[:, i2]) + 1.0, axis=1)
        out["F"] = np.column_stack([f1, f2])


class BT9(_BaseBT):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=3, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        w = np.random.default_rng(1).random((max(1, int(n_pareto_points)), 3))
        w = w / np.linalg.norm(w, axis=1, keepdims=True)
        return w

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        i1 = np.arange(2, d, 3)
        i2 = np.arange(3, d, 3)
        i3 = np.arange(4, d, 3)

        idx = np.arange(1, d + 1, dtype=float)
        y = x - np.sin(np.tile(idx, (n, 1)) * np.pi / (2.0 * d))

        f1 = np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1]) + np.sum(_bt_bias(y[:, i1], 1e-9), axis=1)
        f2 = np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1]) + np.sum(_bt_bias(y[:, i2], 1e-9), axis=1)
        f3 = np.sin(0.5 * np.pi * x[:, 0]) + np.sum(_bt_bias(y[:, i3], 1e-9), axis=1)
        out["F"] = np.column_stack([f1, f2, f3])


for _name in [f"BT{i}" for i in range(1, 10)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"BT{i}" for i in range(1, 10)),
    *(f"BT{i}_JAX" for i in range(1, 10)),
]
