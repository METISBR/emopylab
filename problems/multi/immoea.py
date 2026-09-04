from __future__ import annotations

"""
IMMOEA benchmark family.

Reference
---------
R. Cheng, Y. Jin, K. Narukawa, and B. Sendhoff.
A multiobjective evolutionary algorithm using Gaussian process-based inverse modeling.
IEEE Transactions on Evolutionary Computation, 2015, 19(6): 838-856.
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


class _BaseIMMOEA(Problem):
    _IDX = 1

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        idx = self._IDX
        m_default = 3 if idx in {4, 8} else 2
        m = m_default if n_obj is None else int(n_obj)
        m = max(2, m)
        d = 30 if n_var is None else int(n_var)
        d = max(m, d)
        super().__init__(n_var=d, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _eval_t_a(self, x: np.ndarray) -> np.ndarray:
        d = self.n_var
        j = np.arange(2, d + 1, dtype=float)
        coef = 1.0 + 5.0 * j / d
        return coef[None, :] * x[:, 1:] - x[:, [0]]

    def _eval_t_b(self, x: np.ndarray) -> np.ndarray:
        d = self.n_var
        j = np.arange(2, d + 1, dtype=float)
        exp = 1.0 / (1.0 + 3.0 * j / d)
        return x[:, 1:] ** exp[None, :] - x[:, [0]]

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.clip(_as_2d(x), self.xl, self.xu)
        idx = self._IDX
        d = self.n_var

        if idx in {1, 2, 3}:
            t = self._eval_t_a(x)
            g = 1.0 + 9.0 * np.mean(t**2, axis=1)
            f1 = x[:, 0] if idx != 3 else (1.0 - np.exp(-4.0 * x[:, 0]) * np.sin(6.0 * np.pi * x[:, 0]) ** 6)
            f2 = g * (1.0 - np.sqrt(f1 / g)) if idx == 1 else g * (1.0 - (f1 / g) ** 2)
            out["F"] = np.column_stack([f1, f2])
            return

        if idx == 4:
            t = (1.0 + 5.0 * np.arange(3, d + 1, dtype=float) / d)[None, :] * x[:, 2:] - x[:, [0]]
            g = np.sum(t**2, axis=1)
            f1 = np.cos(np.pi / 2.0 * x[:, 0]) * np.cos(np.pi / 2.0 * x[:, 1]) * (1.0 + g)
            f2 = np.cos(np.pi / 2.0 * x[:, 0]) * np.sin(np.pi / 2.0 * x[:, 1]) * (1.0 + g)
            f3 = np.sin(np.pi / 2.0 * x[:, 0]) * (1.0 + g)
            out["F"] = np.column_stack([f1, f2, f3])
            return

        if idx in {5, 6, 7}:
            t = self._eval_t_b(x)
            g = 1.0 + 9.0 * np.mean(t**2, axis=1)
            f1 = x[:, 0] if idx != 7 else (1.0 - np.exp(-4.0 * x[:, 0]) * np.sin(6.0 * np.pi * x[:, 0]) ** 6)
            f2 = g * (1.0 - np.sqrt(f1 / g)) if idx == 5 else g * (1.0 - (f1 / g) ** 2)
            out["F"] = np.column_stack([f1, f2])
            return

        if idx == 8:
            t = x[:, 2:] ** (1.0 / (1.0 + 3.0 * np.arange(3, d + 1, dtype=float) / d))[None, :] - x[:, [0]]
            g = np.sum(t**2, axis=1)
            f1 = np.cos(np.pi / 2.0 * x[:, 0]) * np.cos(np.pi / 2.0 * x[:, 1]) * (1.0 + g)
            f2 = np.cos(np.pi / 2.0 * x[:, 0]) * np.sin(np.pi / 2.0 * x[:, 1]) * (1.0 + g)
            f3 = np.sin(np.pi / 2.0 * x[:, 0]) * (1.0 + g)
            out["F"] = np.column_stack([f1, f2, f3])
            return

        if idx == 9:
            t = self._eval_t_b(x)
            g = np.sum(t**2 / 4000.0, axis=1) - np.prod(np.cos(t / np.sqrt(np.arange(1, d, dtype=float))[None, :]), axis=1) + 2.0
            f1 = x[:, 0]
            f2 = g * (1.0 - np.sqrt(f1 / g))
            out["F"] = np.column_stack([f1, f2])
            return

        if idx == 10:
            t = self._eval_t_b(x)
            g = 1.0 + 10.0 * (d - 1) + np.sum(t**2 - 10.0 * np.cos(2.0 * np.pi * t), axis=1)
            f1 = x[:, 0]
            f2 = g * (1.0 - np.sqrt(f1 / g))
            out["F"] = np.column_stack([f1, f2])
            return

        raise RuntimeError(f"Unsupported IMMOEA index: {idx}")

    def _calc_pareto_front(self, n_pareto_points=200):
        idx = self._IDX
        if self.n_obj == 3:
            r = _uniform_simplex(n_pareto_points, 3)
            return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

        if idx in {2, 6}:
            x = np.linspace(0.0, 1.0, max(2, int(n_pareto_points)))
            return np.column_stack([x, 1.0 - x**2])

        if idx in {3, 7}:
            x = np.linspace(_MINF1, 1.0, max(2, int(n_pareto_points)))
            return np.column_stack([x, 1.0 - x**2])

        x = np.linspace(0.0, 1.0, max(2, int(n_pareto_points)))
        return np.column_stack([x, 1.0 - np.sqrt(x)])


class IMMOEA_F1(_BaseIMMOEA):
    _IDX = 1


class IMMOEA_F2(_BaseIMMOEA):
    _IDX = 2


class IMMOEA_F3(_BaseIMMOEA):
    _IDX = 3


class IMMOEA_F4(_BaseIMMOEA):
    _IDX = 4


class IMMOEA_F5(_BaseIMMOEA):
    _IDX = 5


class IMMOEA_F6(_BaseIMMOEA):
    _IDX = 6


class IMMOEA_F7(_BaseIMMOEA):
    _IDX = 7


class IMMOEA_F8(_BaseIMMOEA):
    _IDX = 8


class IMMOEA_F9(_BaseIMMOEA):
    _IDX = 9


class IMMOEA_F10(_BaseIMMOEA):
    _IDX = 10


_CPU = [f"IMMOEA_F{i}" for i in range(1, 11)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
