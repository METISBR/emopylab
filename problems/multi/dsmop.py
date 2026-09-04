from __future__ import annotations

"""
DSMOP dynamic sparse benchmark family.

Reference
---------
P. Zhang, R. Zhang, Y. Tian, K. C. Tan, and X. Zhang.
A dual model-based evolutionary framework for dynamic large-scale sparse multiobjective optimization.
Swarm and Evolutionary Computation, 2025, 97: 102011.
"""

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


def _map_linear(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), x[:, : m - 1]]), axis=1))
    b = np.column_stack([np.ones((n, 1)), 1.0 - x[:, m - 2 :: -1]])
    return a * b


def _map_one_minus(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), 1.0 - np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
    b = np.column_stack([np.ones((n, 1)), 1.0 - np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


def _map_cos_sin(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
    b = np.column_stack([np.ones((n, 1)), np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


def _front_one_minus(n: int, m: int):
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
    return _map_one_minus(x, m)


def _g1(x, t):
    return (x - t) ** 2


def _g2(x, t):
    return 2.0 * (x - t) ** 2 + np.sin(2.0 * np.pi * (x - t)) ** 2


def _g3(x, t):
    return 4.0 - (x - t) - 4.0 / np.exp(100.0 * (x - t) ** 2)


def _g4(x, t):
    return (x - np.pi / 3.0) ** 2 + t * np.sin(6.0 * np.pi * (x - np.pi / 3.0)) ** 2


def _sparse_filter_sorted(g_raw, x_tail, k):
    if g_raw.size == 0:
        return np.zeros(g_raw.shape[0])
    order = np.argsort(g_raw, axis=1)
    g_sorted = np.take_along_axis(g_raw, order, axis=1)
    x_sorted = np.take_along_axis(x_tail, order, axis=1)
    mask = x_sorted == 0
    if k > 0:
        mask[:, : min(k, mask.shape[1])] = False
    g_sorted[mask] = 0.0
    return np.sum(g_sorted, axis=1)


class _BaseDSMOP(Problem):
    def __init__(self, *, n_var: int = 100, n_obj: int = 2, theta: float = 0.1, nt: int = 10, taut: int = 10, n_pop: int = 100, **kwargs):
        self.theta = float(theta)
        self.nt = int(nt)
        self.taut = int(taut)
        self.n_pop = int(max(1, n_pop))
        self._n_eval = 0
        self._last_env = 0

        n_var = int(n_var)
        n_obj = int(n_obj)
        self._L = max(0, n_var - n_obj + 1)
        self._K = int(np.ceil(self.theta * self._L))

        xl = np.concatenate([np.zeros(max(0, n_obj - 1)), -np.ones(max(0, n_var - n_obj + 1))])
        xu = np.concatenate([np.ones(max(0, n_obj - 1)), 2.0 * np.ones(max(0, n_var - n_obj + 1))])
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)

    def _env(self):
        T = int(np.floor(self._n_eval / (self.n_pop * self.taut)))
        t = T / self.nt
        return T, t

    def _advance(self, n):
        self._n_eval += int(n)

    def _fixed_k(self):
        return int(np.ceil(self.theta * self._L))

    def _dynamic_k(self, T):
        if T > self._last_env:
            self._last_env = T
            self._K = int(np.ceil(np.random.rand() * self._L * self.theta))
        return int(self._K)

    def _scale(self, g):
        den = max(1, self._L)
        return 1.0 + g / den


class DSMOP1(_BaseDSMOP):
    def __init__(self, **kwargs):
        super().__init__(taut=5, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        _, t = self._env()
        k = self._fixed_k()

        xa = x[:, self.n_obj - 1 : self.n_obj - 1 + k]
        xb = x[:, self.n_obj - 1 + k :]
        g = np.sum(_g1(xa, 0.75 * np.cos(np.pi * t)), axis=1) + np.sum(_g2(xb, 0.0), axis=1)

        out["F"] = self._scale(g)[:, None] * _map_linear(x, self.n_obj)
        self._advance(x.shape[0])


class DSMOP2(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        _, t = self._env()
        k = self._fixed_k()

        xa = x[:, self.n_obj - 1 : self.n_obj - 1 + k]
        xb = x[:, self.n_obj - 1 + k :]
        g = np.sum(_g2(xa, 1.25 - 0.75 * np.abs(np.sin(0.5 * np.pi * t))), axis=1) + np.sum(_g3(xb, 0.0), axis=1)

        out["F"] = self._scale(g)[:, None] * _map_linear(x, self.n_obj)
        self._advance(x.shape[0])


class DSMOP3(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        _, t = self._env()
        k = self._fixed_k()

        xa = x[:, self.n_obj - 1 : self.n_obj - 1 + k]
        xb = x[:, self.n_obj - 1 + k :]
        g = np.sum(_g1(xa, 0.25 + 0.75 * np.sin(0.5 * np.pi * t)), axis=1)

        for s in range(0, xb.shape[1], 10):
            block = xb[:, s : s + 10]
            temp = 50.0 - np.sum(_g1(block, 0.0), axis=1)
            g += np.where(temp < 50.0, temp, 0.0)

        out["F"] = self._scale(g)[:, None] * _map_linear(x, self.n_obj)
        self._advance(x.shape[0])


class DSMOP4(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        T, t = self._env()
        k = self._dynamic_k(T)

        xa = x[:, self.n_obj - 1 : self.n_obj - 1 + k]
        xb = x[:, self.n_obj - 1 + k :]
        if xb.shape[1] > 0:
            tb = np.hstack([xb[:, 1:], xb[:, [0]]]) * 0.9
            gb = np.sum(_g2(xb, tb), axis=1)
        else:
            gb = np.zeros(x.shape[0])
        ga = np.sum(_g1(xa, np.sin(0.5 * np.pi * t)), axis=1)
        g = ga + gb

        out["F"] = self._scale(g)[:, None] * _map_cos_sin(x, self.n_obj)
        self._advance(x.shape[0])


class DSMOP5(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        return _front_one_minus(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        _, t = self._env()
        k = self._fixed_k()

        xb = x[:, self.n_obj - 1 :]
        lin = np.linspace(0.0, 1.0, xb.shape[1])[None, :]
        g_raw = _g4(xb, lin * np.cos(t))
        g = _sparse_filter_sorted(g_raw, xb, k)

        out["F"] = self._scale(g)[:, None] * _map_one_minus(x, self.n_obj)
        self._advance(x.shape[0])


class DSMOP6(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        T, _ = self._env()
        k = self._dynamic_k(T)

        xa = x[:, self.n_obj - 1 : self.n_obj - 1 + k]
        xb = x[:, self.n_obj - 1 + k :]
        g = np.sum(_g2(xa, np.pi / 3.0), axis=1) + np.sum(_g3(xb, 0.0), axis=1)

        out["F"] = self._scale(g)[:, None] * _map_linear(x, self.n_obj)
        self._advance(x.shape[0])


class DSMOP7(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        T, t = self._env()
        k = self._dynamic_k(T)

        xa = x[:, self.n_obj - 1 : self.n_obj - 1 + k]
        xb = x[:, self.n_obj - 1 + k :]
        g = np.sum(_g1(xa, 0.75 * np.cos(np.pi * t)), axis=1) + np.sum(_g2(xb, 0.0), axis=1)

        out["F"] = self._scale(g)[:, None] * _map_linear(x, self.n_obj)
        self._advance(x.shape[0])


class DSMOP8(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        T, t = self._env()
        k = self._dynamic_k(T)

        xa = x[:, self.n_obj - 1 : self.n_obj - 1 + k]
        xb = x[:, self.n_obj - 1 + k :]
        g = np.sum(_g1(xa, 0.25 + 0.75 * np.sin(0.5 * np.pi * t)), axis=1)

        for s in range(0, xb.shape[1], 10):
            block = xb[:, s : s + 10]
            temp = 50.0 - np.sum(_g1(block, 0.0), axis=1)
            g += np.where(temp < 50.0, temp, 0.0)

        out["F"] = self._scale(g)[:, None] * _map_linear(x, self.n_obj)
        self._advance(x.shape[0])


class DSMOP9(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        _, t = self._env()
        p = 1.25 - 0.25 * np.sin(0.5 * np.pi * t)
        return _front_one_minus(n_pareto_points, self.n_obj) ** p

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        _, t = self._env()
        k = self._fixed_k()

        xb = x[:, self.n_obj - 1 :]
        g = np.sum(_g1(xb, np.pi / 3.0) * _g2(xb, 0.0), axis=1) + np.abs(k - np.sum(xb != 0.0, axis=1))

        p = 1.25 - 0.25 * np.sin(0.5 * np.pi * t)
        h = _map_one_minus(x, self.n_obj)
        out["F"] = self._scale(g)[:, None] * (h**p)
        self._advance(x.shape[0])


class DSMOP10(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        _, t = self._env()
        p = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return r**p

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        _, t = self._env()
        k = self._fixed_k()

        xb = x[:, self.n_obj - 1 :]
        gs = np.sort(_g3(xb, 0.0), axis=1)
        keep = max(0, xb.shape[1] - k)
        g = np.sum(gs[:, :keep], axis=1)

        p = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        h = _map_cos_sin(x, self.n_obj)
        out["F"] = self._scale(g)[:, None] * (h**p)
        self._advance(x.shape[0])


class DSMOP11(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        _, t = self._env()
        p = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        return _front_one_minus(n_pareto_points, self.n_obj) ** p

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        T, t = self._env()
        k = self._dynamic_k(T)

        xb = x[:, self.n_obj - 1 :]
        g = np.sum(_g1(xb, np.pi / 3.0) * _g2(xb, 0.0), axis=1) + np.abs(k - np.sum(xb != 0.0, axis=1))

        p = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        h = _map_one_minus(x, self.n_obj)
        out["F"] = self._scale(g)[:, None] * (h**p)
        self._advance(x.shape[0])


class DSMOP12(_BaseDSMOP):
    def _calc_pareto_front(self, n_pareto_points=200):
        _, t = self._env()
        p = 1.25 - 0.25 * np.sin(0.5 * np.pi * t)
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return r**p

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        T, t = self._env()
        k = self._dynamic_k(T)

        xb = x[:, self.n_obj - 1 :]
        lin = np.linspace(0.0, 1.0, xb.shape[1])[None, :]
        g_raw = _g4(xb, lin)
        g = _sparse_filter_sorted(g_raw, xb, k)

        p = 1.25 - 0.25 * np.sin(0.5 * np.pi * t)
        h = _map_cos_sin(x, self.n_obj)
        out["F"] = self._scale(g)[:, None] * (h**p)
        self._advance(x.shape[0])


for _name in [f"DSMOP{i}" for i in range(1, 13)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"DSMOP{i}" for i in range(1, 13)),
    *(f"DSMOP{i}_JAX" for i in range(1, 13)),
]
