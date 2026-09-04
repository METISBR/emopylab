from __future__ import annotations

"""
GLSMOP benchmark family.

Reference
---------
H. Gu, H. Wang, and J. Ma.
Test suites and performance of algorithms in large-scale multiobjective evolutionary optimization.
Proceedings of the IEEE Congress on Evolutionary Computation, 2024.
"""

import numpy as np
from core.problem import Problem


def _uniform_simplex(n: int, m: int, seed: int = 1) -> np.ndarray:
    n = max(2, int(n))
    m = max(2, int(m))
    if m == 2:
        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([x, 1.0 - x])
    rng = np.random.default_rng(seed)
    w = rng.random((n, m))
    w /= np.maximum(np.sum(w, axis=1, keepdims=True), 1e-30)
    return w


def _uniform_grid_box(n: int, dim: int) -> np.ndarray:
    n = max(2, int(n))
    dim = max(1, int(dim))
    if dim == 1:
        return np.linspace(0.0, 1.0, n)[:, None]
    if dim == 2:
        s = int(np.ceil(np.sqrt(n)))
        x = np.linspace(0.0, 1.0, s)
        xx, yy = np.meshgrid(x, x)
        return np.column_stack([xx.ravel(), yy.ravel()])
    rng = np.random.default_rng(1)
    return rng.random((n, dim))


def _nondominated_mask(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return keep


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def _griewank(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    den = np.sqrt(np.arange(1, x.shape[1] + 1, dtype=float))[None, :]
    return np.sum(x**2, axis=1) / 4000.0 - np.prod(np.cos(x / den), axis=1) + 1.0


def _schwefel(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    return np.max(np.abs(x), axis=1)


def _rastrigin(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)


def _rosenbrock(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1:
        return np.zeros(x.shape[0], dtype=float)
    return np.sum(100.0 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def _ackley(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    n = x.shape[1]
    return 20.0 - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / n)) - np.exp(np.sum(np.cos(2.0 * np.pi * x), axis=1) / n) + np.e


class _BaseGLSMOP(Problem):
    _IDX = 1

    def __init__(self, n_obj: int = 3, n_var: int | None = None, nk: int = 5, **kwargs):
        self.nk = max(1, int(nk))
        n_obj = max(2, int(n_obj))
        if n_var is None:
            n_var = 100 * n_obj
        n_var = max(n_obj + 1, int(n_var))
        self._build_grouping(n_obj, n_var)

        xl = np.zeros(n_var, dtype=float)
        xu = np.concatenate([np.ones(max(0, n_obj - 1)), 10.0 * np.ones(max(0, n_var - n_obj + 1))])
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _build_grouping(self, m: int, d: int) -> None:
        c = [3.8 * 0.1 * (1 - 0.1)]
        for _ in range(m - 1):
            c.append(3.8 * c[-1] * (1.0 - c[-1]))
        c = np.asarray(c, dtype=float)
        sublen = np.floor(c / np.sum(c) * (d - m + 1) / self.nk).astype(int)
        self.sublen = sublen
        self.lengths = np.concatenate([[0], np.cumsum(sublen * self.nk)])

    def _clip(self, x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        return np.clip(arr, self.xl, self.xu)

    def _shape_linear(self, x):
        m = self.n_obj
        a = np.fliplr(np.cumprod(np.column_stack([np.ones((x.shape[0], 1)), x[:, : m - 1]]), axis=1))
        b = np.column_stack([np.ones((x.shape[0], 1)), 1.0 - x[:, m - 2 :: -1]])
        return a * b

    def _shape_spherical(self, x):
        m = self.n_obj
        a = np.fliplr(np.cumprod(np.column_stack([np.ones((x.shape[0], 1)), np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
        b = np.column_stack([np.ones((x.shape[0], 1)), np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
        return a * b

    def _get_funcs(self):
        if self._IDX == 1:
            return _sphere, _sphere
        if self._IDX == 2:
            return _griewank, _schwefel
        if self._IDX == 3:
            return _rastrigin, _rosenbrock
        if self._IDX == 4:
            return _ackley, _griewank
        if self._IDX == 5:
            return _sphere, _sphere
        if self._IDX == 6:
            return _rosenbrock, _schwefel
        if self._IDX == 7:
            return _ackley, _rosenbrock
        if self._IDX == 8:
            return _griewank, _sphere
        if self._IDX == 9:
            return _sphere, _ackley
        raise RuntimeError(f"Unsupported GLSMOP index: {self._IDX}")

    def _transform_tail(self, x):
        n, d = x.shape
        m = self.n_obj
        z = x.copy()
        flip_idx = np.arange(m - 1, d, 2, dtype=int)
        z[:, flip_idx] = self.xl[flip_idx][None, :] + self.xu[flip_idx][None, :] - z[:, flip_idx]

        if self._IDX <= 4:
            coeff = 1.0 + (np.arange(m, d + 1, dtype=float) / d)
        else:
            coeff = 1.0 + np.cos(np.arange(m, d + 1, dtype=float) / d * np.pi / 2.0)
        z[:, m - 1 :] = coeff[None, :] * z[:, m - 1 :] - z[:, [0]] * 10.0
        return z

    def _calc_g_matrix(self, z):
        n = z.shape[0]
        m = self.n_obj
        odd_func, even_func = self._get_funcs()
        g = np.zeros((n, m), dtype=float)

        for i in range(m):
            func = odd_func if (i % 2 == 0) else even_func
            for j in range(self.nk):
                start = int(self.lengths[i] + (m - 1) + j * self.sublen[i])
                end = int(self.lengths[i] + (m - 1) + (j + 1) * self.sublen[i])
                start = min(start, self.n_var)
                end = min(end, self.n_var)
                if end <= start:
                    continue
                g[:, i] += func(z[:, start:end])

        den = np.maximum(self.sublen.astype(float), 1.0)[None, :] * float(self.nk)
        return g / den

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        z = self._transform_tail(x)
        g = self._calc_g_matrix(z)
        n = x.shape[0]
        m = self.n_obj

        if self._IDX <= 4:
            f = (1.0 + g) * self._shape_linear(x)
        elif self._IDX <= 8:
            f = (1.0 + g + np.hstack([g[:, 1:], np.zeros((n, 1), dtype=float)])) * self._shape_spherical(x)
        else:
            gs = 1.0 + np.sum(g, axis=1)
            f = np.zeros((n, m), dtype=float)
            f[:, : m - 1] = x[:, : m - 1]
            term = np.sum((f[:, : m - 1] / (1.0 + gs)[:, None]) * (1.0 + np.sin(3.0 * np.pi * f[:, : m - 1])), axis=1)
            f[:, m - 1] = (1.0 + gs) * (m - term)
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=200):
        n = max(20, int(n_pareto_points))
        m = self.n_obj
        if self._IDX <= 4:
            return _uniform_simplex(n, m)
        if self._IDX <= 8:
            r = _uniform_simplex(n, m)
            return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

        x = _uniform_grid_box(n, m - 1)
        interval = np.array([0.0, 0.251412, 0.631627, 0.859401], dtype=float)
        median = (interval[1] - interval[0]) / ((interval[3] - interval[2]) + (interval[1] - interval[0]))
        left = x <= median
        x[left] = x[left] * (interval[1] - interval[0]) / median + interval[0]
        x[~left] = (x[~left] - median) * (interval[3] - interval[2]) / (1.0 - median) + interval[2]
        f_last = 2.0 * (m - np.sum(x / 2.0 * (1.0 + np.sin(3.0 * np.pi * x)), axis=1))
        f = np.column_stack([x, f_last])
        mask = _nondominated_mask(f)
        return f[mask]


class GLSMOP1(_BaseGLSMOP):
    _IDX = 1


class GLSMOP2(_BaseGLSMOP):
    _IDX = 2


class GLSMOP3(_BaseGLSMOP):
    _IDX = 3


class GLSMOP4(_BaseGLSMOP):
    _IDX = 4


class GLSMOP5(_BaseGLSMOP):
    _IDX = 5


class GLSMOP6(_BaseGLSMOP):
    _IDX = 6


class GLSMOP7(_BaseGLSMOP):
    _IDX = 7


class GLSMOP8(_BaseGLSMOP):
    _IDX = 8


class GLSMOP9(_BaseGLSMOP):
    _IDX = 9


_CPU = [f"GLSMOP{i}" for i in range(1, 10)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
