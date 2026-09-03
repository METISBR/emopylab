from __future__ import annotations

"""
RMMEDA benchmark family.

Reference
---------
Q. Zhang, A. Zhou, and Y. Jin.
RM-MEDA: A regularity model-based multiobjective estimation of distribution algorithm.
IEEE Transactions on Evolutionary Computation, 2008, 12(1): 41-63.
"""

import numpy as np

from core.problem import Problem


_MINF1 = 0.2807753191


def _as_2d(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _uniform_simplex(n: int, m: int, seed: int = 1) -> np.ndarray:
    n = max(2, int(n))
    if m == 2:
        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([x, 1.0 - x])
    rng = np.random.default_rng(seed)
    w = rng.random((n, m))
    return w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-30)


class _BaseRMMEDA(Problem):
    _IDX = 1

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        idx = self._IDX
        m_default = 3 if idx in {4, 8} else 2
        m = m_default if n_obj is None else int(n_obj)
        m = max(2, m)
        d = 30 if n_var is None else int(n_var)
        d = max(m, d)
        super().__init__(n_var=d, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.clip(_as_2d(x), self.xl, self.xu)
        idx = self._IDX
        d = self.n_var

        if idx in {1, 2, 3}:
            t = x[:, 1:] - x[:, [0]]
            if idx == 3:
                g = 1.0 + 9.0 * (np.sum(t**2, axis=1) / 9.0) ** 0.25
                f1 = 1.0 - np.exp(-4.0 * x[:, 0]) * np.sin(6.0 * np.pi * x[:, 0]) ** 6
                f2 = g * (1.0 - (f1 / g) ** 2)
            else:
                g = 1.0 + 9.0 * np.mean(t**2, axis=1)
                f1 = x[:, 0]
                f2 = g * (1.0 - np.sqrt(f1 / g)) if idx == 1 else g * (1.0 - (f1 / g) ** 2)
            out["F"] = np.column_stack([f1, f2])
            return

        if idx == 4:
            g = np.sum((x[:, 2:] - x[:, [0]]) ** 2, axis=1)
            f1 = np.cos(np.pi / 2.0 * x[:, 0]) * np.cos(np.pi / 2.0 * x[:, 1]) * (1.0 + g)
            f2 = np.cos(np.pi / 2.0 * x[:, 0]) * np.sin(np.pi / 2.0 * x[:, 1]) * (1.0 + g)
            f3 = np.sin(np.pi / 2.0 * x[:, 0]) * (1.0 + g)
            out["F"] = np.column_stack([f1, f2, f3])
            return

        if idx in {5, 6, 7}:
            t = x[:, 1:] ** 2 - x[:, [0]]
            if idx == 7:
                g = 1.0 + 9.0 * (np.sum(t**2, axis=1) / 9.0) ** 0.25
                f1 = 1.0 - np.exp(-4.0 * x[:, 0]) * np.sin(6.0 * np.pi * x[:, 0]) ** 6
                f2 = g * (1.0 - (f1 / g) ** 2)
            else:
                g = 1.0 + 9.0 * np.mean(t**2, axis=1)
                f1 = x[:, 0] if idx == 5 else np.sqrt(x[:, 0])
                f2 = g * (1.0 - np.sqrt(f1 / g)) if idx == 5 else g * (1.0 - (f1 / g) ** 2)
            out["F"] = np.column_stack([f1, f2])
            return

        if idx == 8:
            g = np.sum((x[:, 2:] ** 2 - x[:, [0]]) ** 2, axis=1)
            f1 = np.cos(np.pi / 2.0 * x[:, 0]) * np.cos(np.pi / 2.0 * x[:, 1]) * (1.0 + g)
            f2 = np.cos(np.pi / 2.0 * x[:, 0]) * np.sin(np.pi / 2.0 * x[:, 1]) * (1.0 + g)
            f3 = np.sin(np.pi / 2.0 * x[:, 0]) * (1.0 + g)
            out["F"] = np.column_stack([f1, f2, f3])
            return

        t = x[:, 1:] ** 2 - x[:, [0]]
        if idx == 9:
            g = np.sum(t**2 / 4000.0, axis=1) - np.prod(np.cos(t / np.sqrt(np.arange(1, d, dtype=float))[None, :]), axis=1) + 2.0
        else:  # idx == 10
            g = 1.0 + 10.0 * (d - 1) + np.sum(t**2 - 10.0 * np.cos(2.0 * np.pi * t), axis=1)
        f1 = x[:, 0]
        f2 = g * (1.0 - np.sqrt(f1 / g))
        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, n_pareto_points=200):
        idx = self._IDX
        n = max(2, int(n_pareto_points))

        if self.n_obj == 3:
            r = _uniform_simplex(n, 3)
            return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

        if idx in {2, 6}:
            x = np.linspace(0.0, 1.0, n)
            return np.column_stack([x, 1.0 - x**2])

        if idx in {3, 7}:
            x = np.linspace(_MINF1, 1.0, n)
            return np.column_stack([x, 1.0 - x**2])

        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([x, 1.0 - np.sqrt(x)])


class RMMEDA_F1(_BaseRMMEDA):
    _IDX = 1


class RMMEDA_F2(_BaseRMMEDA):
    _IDX = 2


class RMMEDA_F3(_BaseRMMEDA):
    _IDX = 3


class RMMEDA_F4(_BaseRMMEDA):
    _IDX = 4


class RMMEDA_F5(_BaseRMMEDA):
    _IDX = 5


class RMMEDA_F6(_BaseRMMEDA):
    _IDX = 6


class RMMEDA_F7(_BaseRMMEDA):
    _IDX = 7


class RMMEDA_F8(_BaseRMMEDA):
    _IDX = 8


class RMMEDA_F9(_BaseRMMEDA):
    _IDX = 9


class RMMEDA_F10(_BaseRMMEDA):
    _IDX = 10


_CPU = [f"RMMEDA_F{i}" for i in range(1, 11)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
