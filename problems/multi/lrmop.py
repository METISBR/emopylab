from __future__ import annotations

"""
LRMOP benchmark family.

Reference
---------
S. Shao, Y. Tian, L. Zhang, K. C. Tan, and X. Zhang.
An evolutionary algorithm for solving large-scale robust multi-objective optimization problems.
IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2476-2490.
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


def _g1(x, t):
    return (3.0 + x - 4.0 * t**3) ** 2


def _g2(x, t):
    return 4.0 - (x - t) - 4.0 / np.exp(100.0 * (x - t) ** 2)


def _g3(x, t):
    return 4.0 - (x - t) - 4.0 / np.exp(100.0 * (x - t) ** 2)


def _g4(x, t):
    return 2.0 * (x - t) ** 2 + np.sin(2.0 * np.pi * (x - t)) ** 2


def _g5(x, t):
    return np.exp(np.log(2.0) * (x - t) ** 2) * (np.sin(6.0 * np.pi * (x - t)) ** 2) + (x - t) ** 2


def _map_linear(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), x[:, : m - 1]]), axis=1))
    b = np.column_stack([np.ones((n, 1)), 1.0 - x[:, m - 2 :: -1]])
    return a * b


def _map_1mcos_1msin(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    u = 1.0 - np.cos(x[:, : m - 1] * np.pi / 2.0)
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), u]), axis=1))
    b = np.column_stack([np.ones((n, 1)), 1.0 - np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


def _map_cos_sin(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    u = np.cos(x[:, : m - 1] * np.pi / 2.0)
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), u]), axis=1))
    b = np.column_stack([np.ones((n, 1)), np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


def _optimum_transform_lrmop234(r, m):
    c = np.ones((r.shape[0], m), dtype=float)
    for i in range(r.shape[0]):
        for j in range(2, m + 1):
            start = m - j + 1
            end = m - 1
            prod_term = np.prod(1.0 - c[i, start:end]) if end > start else 1.0
            temp = r[i, j - 1] / max(r[i, 0], 1e-30) * prod_term
            c[i, m - j] = (temp**2 - temp + np.sqrt(max(0.0, 2.0 * temp))) / (temp**2 + 1.0)
    x = np.arccos(np.clip(c, -1.0, 1.0)) * 2.0 / np.pi
    return _map_1mcos_1msin(x, m)


class _BaseLRMOP(Problem):
    def __init__(self, n_var: int = 100, n_obj: int = 2, theta: float = 0.1, h: int = 50, **kwargs):
        self.theta = float(theta)
        self.H = int(h)
        n_var = int(n_var)
        n_obj = int(n_obj)
        xl = np.concatenate([np.zeros(max(0, n_obj - 1)), -np.ones(max(0, n_var - n_obj + 1))])
        xu = np.concatenate([np.ones(max(0, n_obj - 1)), 2.0 * np.ones(max(0, n_var - n_obj + 1))])
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)

    def _scale(self, g):
        den = max(1, self.n_var - self.n_obj + 1)
        return 1.0 + g / den


class LRMOP1(_BaseLRMOP):
    def _calc_pareto_front(self, n_pareto_points=100):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = int(np.ceil(self.theta * (self.n_var - self.n_obj + 1)))
        K = int(k + np.round(k * np.random.rand()))
        a = np.random.normal(1.0, 0.2)
        g = np.sum(_g1(x[:, 1:], a) * _g2(x[:, 1:], 0.0), axis=1) + np.abs(K - np.sum(x[:, 1:] != 0.0, axis=1))

        f = self._scale(g)[:, None] * _map_linear(x, self.n_obj)
        out["F"] = f


class LRMOP2(_BaseLRMOP):
    def _calc_pareto_front(self, n_pareto_points=100):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return _optimum_transform_lrmop234(r, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = int(np.ceil(self.theta * (self.n_var - self.n_obj + 1)))
        a = np.random.normal(1.0, 0.2)
        g = np.sum(_g1(x[:, 1:], a) * _g3(x[:, 1:], 0.0), axis=1) + np.abs(k - np.sum(x[:, 1:] != 0.0, axis=1))

        f = self._scale(g)[:, None] * _map_1mcos_1msin(x, self.n_obj)
        out["F"] = f


class LRMOP3(_BaseLRMOP):
    def _calc_pareto_front(self, n_pareto_points=100):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return _optimum_transform_lrmop234(r, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = int(np.ceil(self.theta * (self.n_var - self.n_obj + 1)))
        a = np.random.normal(1.0, 0.2)
        g = np.sum(_g1(x[:, 1:], a) * _g5(x[:, 1:], 0.0), axis=1) + np.abs(k - np.sum(x[:, 1:] != 0.0, axis=1))

        f = self._scale(g)[:, None] * _map_1mcos_1msin(x, self.n_obj)
        out["F"] = f


class LRMOP4(_BaseLRMOP):
    def _calc_pareto_front(self, n_pareto_points=100):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return _optimum_transform_lrmop234(r, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = int(np.ceil(self.theta * (self.n_var - self.n_obj + 1)))
        K = int(k + np.round(k * np.random.rand()))
        a = np.random.normal(1.0, 0.2)
        g = np.sum(_g1(x[:, 1:], a) * _g5(x[:, 1:], 0.0), axis=1) + np.abs(K - np.sum(x[:, 1:] != 0.0, axis=1))

        f = self._scale(g)[:, None] * _map_1mcos_1msin(x, self.n_obj)
        out["F"] = f


class LRMOP5(_BaseLRMOP):
    def _calc_pareto_front(self, n_pareto_points=100):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = int(np.ceil(self.theta * (self.n_var - self.n_obj + 1)))
        K = int(k)
        a = np.random.normal(1.0, 0.2)
        g = np.sum(_g1(x[:, 1:], a) * _g2(x[:, 1:], 0.0), axis=1) + np.abs(K - np.sum(x[:, 1:] != 0.0, axis=1))

        f = self._scale(g)[:, None] * _map_cos_sin(x, self.n_obj)
        out["F"] = f


class LRMOP6(_BaseLRMOP):
    def _calc_pareto_front(self, n_pareto_points=100):
        return _uniform_simplex(n_pareto_points, self.n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = int(np.ceil(self.theta * (self.n_var - self.n_obj + 1)))
        K = int(k)
        a = np.random.normal(1.0, 0.2)
        g = np.sum(_g1(x[:, 1:], a) * _g4(x[:, 1:], 0.0), axis=1) + np.abs(K - np.sum(x[:, 1:] != 0.0, axis=1))

        f = self._scale(g)[:, None] * _map_linear(x, self.n_obj)
        out["F"] = f


for _name in [f"LRMOP{i}" for i in range(1, 7)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"LRMOP{i}" for i in range(1, 7)),
    *(f"LRMOP{i}_JAX" for i in range(1, 7)),
]
