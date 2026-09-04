from __future__ import annotations

"""
SMOP benchmark family.

Reference
---------
Y. Tian, X. Zhang, C. Wang, and Y. Jin.
An evolutionary algorithm for large-scale sparse multi-objective optimization problems.
IEEE Transactions on Evolutionary Computation, 2020, 24(2): 380-393.
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


def _pf_type_456(n: int, m: int) -> np.ndarray:
    r = _uniform_simplex(n, m)
    c = np.ones((r.shape[0], m), dtype=float)
    for i in range(r.shape[0]):
        for j in range(2, m + 1):
            start = m - j + 1
            end = m - 1
            prod_term = np.prod(1.0 - c[i, start:end]) if end > start else 1.0
            temp = r[i, j - 1] / max(r[i, 0], 1e-30) * prod_term
            c[i, m - j] = (temp**2 - temp + np.sqrt(max(0.0, 2.0 * temp))) / (temp**2 + 1.0)
    x = np.arccos(np.clip(c, -1.0, 1.0)) * 2.0 / np.pi
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((x.shape[0], 1)), 1.0 - np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
    b = np.column_stack([np.ones((x.shape[0], 1)), 1.0 - np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


def _g1(x, t):
    return (x - t) ** 2


def _g2(x, t):
    return 2.0 * (x - t) ** 2 + np.sin(2.0 * np.pi * (x - t)) ** 2


def _g3(x, t):
    expo = np.minimum(100.0 * (x - t) ** 2, 700.0)
    return 4.0 - (x - t) - 4.0 / np.exp(expo)


def _g4(x, t):
    return (x - np.pi / 3.0) ** 2 + t * np.sin(6.0 * np.pi * (x - np.pi / 3.0)) ** 2


class _BaseSMOP(Problem):
    _IDX = 1

    def __init__(self, n_obj: int = 2, n_var: int = 100, theta: float = 0.1, **kwargs):
        self.theta = float(theta)
        n_obj = max(2, int(n_obj))
        n_var = max(n_obj + 1, int(n_var))

        xl = np.concatenate([np.zeros(max(0, n_obj - 1)), -np.ones(max(0, n_var - n_obj + 1))])
        xu = np.concatenate([np.ones(max(0, n_obj - 1)), 2.0 * np.ones(max(0, n_var - n_obj + 1))])
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

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

    def _shape_type456(self, x):
        m = self.n_obj
        a = np.fliplr(np.cumprod(np.column_stack([np.ones((x.shape[0], 1)), 1.0 - np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
        b = np.column_stack([np.ones((x.shape[0], 1)), 1.0 - np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
        return a * b

    def _shape_type78(self, x):
        m = self.n_obj
        a = np.fliplr(np.cumprod(np.column_stack([np.ones((x.shape[0], 1)), np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
        b = np.column_stack([np.ones((x.shape[0], 1)), np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
        return a * b

    def _calc_g(self, x):
        m = self.n_obj
        d = self.n_var
        tail_len = d - m + 1
        k = int(np.ceil(self.theta * tail_len))
        tail = x[:, m - 1 :]

        if self._IDX == 1:
            g = np.sum(_g1(tail[:, :k], np.pi / 3.0), axis=1) + np.sum(_g2(tail[:, k:], 0.0), axis=1)
        elif self._IDX == 2:
            g = np.sum(_g2(tail[:, :k], np.pi / 3.0), axis=1) + np.sum(_g3(tail[:, k:], 0.0), axis=1)
        elif self._IDX == 3:
            g = np.sum(_g1(tail[:, :k], np.pi / 3.0), axis=1)
        elif self._IDX == 4:
            gvals = np.sort(_g3(tail, 0.0), axis=1)
            keep = max(1, d - m - k + 1)
            g = np.sum(gvals[:, :keep], axis=1)
        elif self._IDX == 5:
            g = np.sum(_g1(tail, np.pi / 3.0) * _g2(tail, 0.0), axis=1) + np.abs(k - np.sum(tail != 0.0, axis=1))
        elif self._IDX == 6:
            t = np.linspace(0.0, 1.0, tail_len)[None, :]
            gvals = _g4(tail, t)
            rank = np.argsort(gvals, axis=1)
            g_sorted = np.take_along_axis(gvals, rank, axis=1)
            temp = np.zeros_like(rank, dtype=bool)
            for i in range(rank.shape[0]):
                idx_dec = (m - 1) + rank[i]
                temp[i] = x[i, idx_dec] == 0.0
            temp[:, :k] = False
            g_sorted[temp] = 0.0
            g = np.sum(g_sorted, axis=1)
        elif self._IDX == 7:
            part1 = np.sum(_g2(tail[:, :k], np.pi / 3.0), axis=1)
            part2_x = tail[:, k - 1 :]
            t2 = np.hstack([x[:, m + k - 1 :], x[:, [m + k - 2]]]) * 0.9
            part2 = np.sum(_g2(part2_x, t2), axis=1)
            g = part1 + part2
        elif self._IDX == 8:
            part1_x = tail[:, :k]
            if k > 0:
                t1_src = x[:, m : m + k]
                if t1_src.shape[1] < k:
                    t1_src = np.hstack([t1_src, np.repeat(x[:, [m + k - 1]], k - t1_src.shape[1], axis=1)])
                t1 = np.mod(t1_src + np.pi, 2.0)
                part1 = np.sum(_g3(part1_x, t1), axis=1)
            else:
                part1 = np.zeros(x.shape[0], dtype=float)

            if tail.shape[1] - k > 0:
                part2_x = tail[:, k:-1]
                t2 = x[:, m + k :] * 0.9
                part2 = np.sum(_g3(part2_x, t2), axis=1)
            else:
                part2 = np.zeros(x.shape[0], dtype=float)
            g = part1 + part2
        else:
            raise RuntimeError(f"Unknown SMOP index: {self._IDX}")

        return g / tail_len

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = self._calc_g(x)

        if self._IDX <= 3:
            shape = self._shape_linear(x)
        elif self._IDX <= 6:
            shape = self._shape_type456(x)
        else:
            shape = self._shape_type78(x)

        out["F"] = (1.0 + g)[:, None] * shape

    def _calc_pareto_front(self, n_pareto_points=200):
        n = max(10, int(n_pareto_points))
        if self._IDX <= 3:
            return _uniform_simplex(n, self.n_obj)
        if self._IDX <= 6:
            return _pf_type_456(n, self.n_obj)
        r = _uniform_simplex(n, self.n_obj)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)


class SMOP1(_BaseSMOP):
    _IDX = 1


class SMOP2(_BaseSMOP):
    _IDX = 2


class SMOP3(_BaseSMOP):
    _IDX = 3


class SMOP4(_BaseSMOP):
    _IDX = 4


class SMOP5(_BaseSMOP):
    _IDX = 5


class SMOP6(_BaseSMOP):
    _IDX = 6


class SMOP7(_BaseSMOP):
    _IDX = 7


class SMOP8(_BaseSMOP):
    _IDX = 8


_CPU = [f"SMOP{i}" for i in range(1, 9)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
