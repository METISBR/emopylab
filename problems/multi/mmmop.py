from __future__ import annotations

"""
MMMOP benchmark family.

Reference
---------
Y. Liu, G. G. Yen, and D. Gong.
A multi-modal multi-objective evolutionary algorithm using two-archive and recombination strategies.
IEEE Transactions on Evolutionary Computation, 2019, 23(4): 660-674.
"""

import itertools
import numpy as np
from core.problem import Problem


def _uniform_simplex(n: int, m: int, seed: int = 1) -> np.ndarray:
    n = max(1, int(n))
    m = max(1, int(m))
    if m == 1:
        return np.ones((n, 1))
    if m == 2:
        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([x, 1.0 - x])
    rng = np.random.default_rng(seed)
    w = rng.random((n, m))
    return w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-30)


def _grid(gap, m):
    m = int(m)
    if m < 1:
        return np.zeros((1, 0))
    vals = np.asarray(gap, dtype=float).ravel().tolist()
    return np.asarray(list(itertools.product(vals, repeat=m)), dtype=float)


def _map_linear(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), x[:, : m - 1]]), axis=1))
    b = np.column_stack([np.ones((n, 1)), 1.0 - x[:, m - 2 :: -1]])
    return a * b


def _map_cos_sin(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
    b = np.column_stack([np.ones((n, 1)), np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


class _BaseMMMOP(Problem):
    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)


class MMMOP1(_BaseMMMOP):
    def __init__(self, n_var: int | None = None, n_obj: int = 2, kA: int = 1, **kwargs):
        m = int(n_obj)
        n = int(m + 1 if n_var is None else n_var)
        self.kA = int(kA)
        super().__init__(n_var=n, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m, d = self.n_obj, self.n_var
        kA = min(self.kA, max(0, d - m + 1))

        xa = x[:, m - 1 : m - 1 + kA]
        xt = x[:, m - 1 + kA :]
        g = 100.0 * (
            d - m + 1
            - np.sum(np.sin(5.0 * np.pi * xa) ** 6, axis=1)
            + np.sum((xt - 0.5) ** 2 - np.cos(20.0 * np.pi * (xt - 0.5)), axis=1)
        )

        f = (1.0 + g)[:, None] * _map_linear(x, m)
        out["F"] = f


class MMMOP2(_BaseMMMOP):
    def __init__(self, n_var: int | None = None, n_obj: int = 2, kA: int = 1, **kwargs):
        m = int(n_obj)
        n = int(m + 1 if n_var is None else n_var)
        self.kA = int(kA)
        super().__init__(n_var=n, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m, d = self.n_obj, self.n_var
        kA = min(self.kA, max(0, d - m + 1))

        xp = x.copy()
        xp[:, : m - 1] = xp[:, : m - 1] ** 100.0

        y = 9.75 * xp[:, m - 1 : m - 1 + kA] + 0.25
        xt = xp[:, m - 1 + kA :]
        g = kA - np.sum(np.sin(10.0 * np.log(np.maximum(y, 1e-30))), axis=1) + np.sum((xt - 0.5) ** 2, axis=1)

        f = (1.0 + g)[:, None] * _map_cos_sin(xp, m)
        out["F"] = f


class MMMOP3(_BaseMMMOP):
    def __init__(self, n_var: int | None = None, n_obj: int = 2, kA: int = 1, c: float = 2.0, d_param: int = 2, **kwargs):
        m = int(n_obj)
        n = int(m + 1 if n_var is None else n_var)
        self.kA = int(kA)
        self.c = float(c)
        self.d_param = int(d_param)
        super().__init__(n_var=n, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m, d = self.n_obj, self.n_var
        kA = min(self.kA, max(0, d - m + 1))

        y = x[:, : m - 1] * self.d_param - np.floor(x[:, : m - 1] * self.d_param)
        xa = x[:, m - 1 : m - 1 + kA]
        xt = x[:, m - 1 + kA :]
        g = kA + np.sum(np.cos(2.0 * np.pi * self.c * xa), axis=1) + np.sum((xt - 0.5) ** 2, axis=1)

        f = (1.0 + g)[:, None] * _map_cos_sin(y, m)
        out["F"] = f


class MMMOP4(_BaseMMMOP):
    def __init__(self, n_var: int | None = None, n_obj: int = 2, kA: int = 1, c: float = 2.0, d_param: int = 2, **kwargs):
        m = int(n_obj)
        n = int(m + 1 if n_var is None else n_var)
        self.kA = int(kA)
        self.c = float(c)
        self.d_param = int(d_param)
        super().__init__(n_var=n, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m, d = self.n_obj, self.n_var
        kA = min(self.kA, max(0, d - m + 1))

        y = x[:, : m - 1] * self.d_param * (self.d_param + 1) / 2.0
        temp = np.cumsum(np.arange(self.d_param, 0, -1, dtype=float))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                idx = int(np.searchsorted(temp, y[i, j], side="left"))
                idx = min(max(idx, 0), self.d_param - 1)
                y[i, j] = (y[i, j] - temp[idx] + (self.d_param - idx)) / self.d_param

        xa = x[:, m - 1 : m - 1 + kA]
        xt = x[:, m - 1 + kA :]
        g = 100.0 * (
            d - m + 1
            + np.sum(np.cos(2.0 * np.pi * self.c * xa), axis=1)
            + np.sum((xt - 0.5) ** 2 - np.cos(20.0 * np.pi * (xt - 0.5)), axis=1)
        )

        f = (1.0 + g)[:, None] * _map_cos_sin(y, m)
        out["F"] = f


class MMMOP5(_BaseMMMOP):
    def __init__(self, n_var: int | None = None, n_obj: int = 2, kA: int = 1, c: float = 2.0, d_param: int = 2, **kwargs):
        m = int(n_obj)
        n = int(m + 1 if n_var is None else n_var)
        self.kA = int(kA)
        self.c = float(c)
        self.d_param = int(d_param)
        super().__init__(n_var=n, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m, d = self.n_obj, self.n_var
        kA = min(self.kA, max(0, d - m + 1))

        y = x[:, : m - 1] * (2.0 ** (self.d_param + 1) - 1.0)
        temp = np.cumsum(2.0 ** np.arange(self.d_param, -1, -1, dtype=float))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                idx = int(np.searchsorted(temp, y[i, j], side="left"))
                idx = min(max(idx, 0), self.d_param)
                denom = 2.0 ** (self.d_param - idx)
                y[i, j] = (y[i, j] - temp[idx] + denom) / denom

        xa = x[:, m - 1 : m - 1 + kA]
        xt = x[:, m - 1 + kA :]
        g = 100.0 * (
            d - m + 1
            + np.sum(np.cos(2.0 * np.pi * self.c * xa), axis=1)
            + np.sum((xt - 0.5) ** 2 - np.cos(20.0 * np.pi * (xt - 0.5)), axis=1)
        )

        f = (1.0 + g)[:, None] * _map_cos_sin(y, m)
        out["F"] = f


class MMMOP6(_BaseMMMOP):
    def __init__(self, n_var: int | None = None, n_obj: int = 2, kA: int = 2, c: float = 2.0, **kwargs):
        m = int(n_obj)
        n = int(m + 2 if n_var is None else n_var)
        self.kA = int(np.ceil(kA / 2.0) * 2)
        self.c = float(c)
        super().__init__(n_var=n, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m, d = self.n_obj, self.n_var
        kA = min(self.kA, max(0, d - m + 1))

        xa = x[:, m - 1 : m - 1 + kA]
        y = 12.0 * (xa - 0.5)

        xb = x[:, m - 1 + kA :]
        z = 2.0 * self.c * xb - 2.0 * np.floor(self.c * xb) - 1.0

        l = d - m - kA + 1
        if l > 0 and m - 1 > 0:
            angles = np.arange(0, l, dtype=float) * np.pi / l
            t = np.prod(np.sin(2.0 * np.pi * x[:, None, : m - 1] + angles[None, :, None]), axis=2)
        else:
            t = np.ones((x.shape[0], max(0, l)))

        y1 = y[:, 0 : kA : 2]
        y2 = y[:, 1 : kA : 2]
        gA = np.sum((y1**2 + y2 - 11.0) ** 2 + (y1 + y2**2 - 7.0) ** 2, axis=1)
        gB = np.sum((z - t) ** 2, axis=1) if z.size > 0 else np.zeros(x.shape[0])
        g = gA + gB

        f = g[:, None] + _map_cos_sin(x, m)
        out["F"] = f


for _name in [f"MMMOP{i}" for i in range(1, 7)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"MMMOP{i}" for i in range(1, 7)),
    *(f"MMMOP{i}_JAX" for i in range(1, 7)),
]
