from __future__ import annotations

"""
SMMOP benchmark family.

Reference
---------
Y. Tian, R. Liu, X. Zhang, H. Ma, K. C. Tan, and Y. Jin.
A multipopulation evolutionary algorithm for solving large-scale multimodal multiobjective optimization problems.
IEEE Transactions on Evolutionary Computation, 2021, 25(3): 405-418.
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
    return np.sqrt((x - t) ** 2) + np.sin(2.0 * np.pi * (x - t)) ** 2


def _g5(x, t):
    return np.exp(np.log(2.0) * ((x - t) ** 2)) * (np.sin(6.0 * np.pi * (x - t)) ** 2) + (x - t) ** 2


class _BaseSMMOP(Problem):
    _IDX = 1

    def __init__(self, n_obj: int = 2, n_var: int = 100, theta: float = 0.1, np_sets: int = 4, **kwargs):
        self.theta = float(theta)
        if "np" in kwargs:
            np_sets = int(kwargs.pop("np"))
        self.np_sets = max(1, int(np_sets))

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

    def _select_g_funcs(self):
        if self._IDX == 1:
            return _g1, _g2
        if self._IDX == 2:
            return _g1, _g3
        if self._IDX == 3:
            return _g2, _g3
        if self._IDX == 4:
            return _g1, _g4
        if self._IDX == 5:
            return _g1, _g5
        if self._IDX == 6:
            return _g2, _g4
        if self._IDX == 7:
            return _g2, _g5
        if self._IDX == 8:
            return _g4, _g5
        raise RuntimeError(f"Unknown SMMOP index: {self._IDX}")

    def _calc_g(self, x):
        n, d = x.shape
        m = self.n_obj
        s = int(np.ceil(self.theta * (d - m)))
        s = max(1, s)
        g_a, g_b = self._select_g_funcs()
        g_all = np.full((n, self.np_sets), np.inf, dtype=float)

        start_tail = m - 1
        tail_len = d - m + 1
        for i in range(self.np_sets):
            seg_start = start_tail + i * s
            seg_end = seg_start + s
            seg_start = min(seg_start, d)
            seg_end = min(seg_end, d)

            seg = x[:, seg_start:seg_end]
            rest_left = x[:, start_tail:seg_start]
            rest_right = x[:, seg_end:]
            rest = np.concatenate([rest_left, rest_right], axis=1)

            g1 = np.sum(g_a(seg, np.pi / 3.0), axis=1) if seg.shape[1] > 0 else 0.0
            g2 = np.sum(g_b(rest, 0.0), axis=1) if rest.shape[1] > 0 else 0.0
            g_all[:, i] = g1 + g2

        g = np.min(g_all, axis=1) / tail_len
        return g

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


class SMMOP1(_BaseSMMOP):
    _IDX = 1


class SMMOP2(_BaseSMMOP):
    _IDX = 2


class SMMOP3(_BaseSMMOP):
    _IDX = 3


class SMMOP4(_BaseSMMOP):
    _IDX = 4


class SMMOP5(_BaseSMMOP):
    _IDX = 5


class SMMOP6(_BaseSMMOP):
    _IDX = 6


class SMMOP7(_BaseSMMOP):
    _IDX = 7


class SMMOP8(_BaseSMMOP):
    _IDX = 8


_CPU = [f"SMMOP{i}" for i in range(1, 9)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
