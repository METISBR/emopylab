from __future__ import annotations

"""
MOEADM2M benchmark family.

Reference
---------
H. Liu, F. Gu, and Q. Zhang.
Decomposition of a multiobjective optimization problem into a number of simple multiobjective subproblems.
IEEE Transactions on Evolutionary Computation, 2014, 18(3): 450-455.
"""

import numpy as np

from core.problem import Problem


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


def _nondominated_mask(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
            continue
        keep &= ~(np.all(f[i] <= f, axis=1) & np.any(f[i] < f, axis=1))
        keep[i] = True
    return keep


class _BaseMOEADM2M(Problem):
    _IDX = 1

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        idx = self._IDX
        m_default = 3 if idx in {6, 7} else 2
        d_default = 10
        m = m_default if n_obj is None else int(n_obj)
        d = d_default if n_var is None else int(n_var)
        d = max(m, d)
        super().__init__(n_var=d, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.clip(_as_2d(x), self.xl, self.xu)
        idx = self._IDX

        if idx in {1, 2, 3, 4, 5}:
            t = x[:, 1:] - np.sin(np.pi / 2.0 * x[:, [0]])

            if idx == 1:
                g = np.sin(np.pi * x[:, 0]) * np.sum(-0.9 * t**2 + np.abs(t) ** 0.6, axis=1)
                f1 = (1.0 + g) * x[:, 0]
                f2 = (1.0 + g) * (1.0 - np.sqrt(x[:, 0]))

            elif idx == 2:
                g = 10.0 * np.sin(np.pi * x[:, 0]) * np.sum(np.abs(t) / (1.0 + np.exp(5.0 * np.abs(t))), axis=1)
                f1 = (1.0 + g) * x[:, 0]
                f2 = (1.0 + g) * (1.0 - x[:, 0] ** 2)

            elif idx == 3:
                g = 10.0 * np.sin(np.pi / 2.0 * x[:, 0]) * np.sum(np.abs(t) / (1.0 + np.exp(5.0 * np.abs(t))), axis=1)
                f1 = (1.0 + g) * np.cos(np.pi / 2.0 * x[:, 0])
                f2 = (1.0 + g) * np.sin(np.pi / 2.0 * x[:, 0])

            elif idx == 4:
                g = 10.0 * np.sin(np.pi * x[:, 0]) * np.sum(np.abs(t) / (1.0 + np.exp(5.0 * np.abs(t))), axis=1)
                f1 = (1.0 + g) * x[:, 0]
                f2 = (1.0 + g) * (1.0 - np.sqrt(x[:, 0]) * np.cos(2.0 * np.pi * x[:, 0]) ** 2)

            else:  # idx == 5
                g = 2.0 * np.abs(np.cos(np.pi * x[:, 0])) * np.sum(-0.9 * t**2 + np.abs(t) ** 0.6, axis=1)
                f1 = (1.0 + g) * x[:, 0]
                f2 = (1.0 + g) * (1.0 - np.sqrt(x[:, 0]))

            out["F"] = np.column_stack([f1, f2])
            return

        t = x[:, 2:] - (x[:, [0]] * x[:, [1]])
        g = 2.0 * np.sin(np.pi * x[:, 0]) * np.sum(-0.9 * t**2 + np.abs(t) ** 0.6, axis=1)

        if idx == 6:
            f1 = (1.0 + g) * x[:, 0] * x[:, 1]
            f2 = (1.0 + g) * x[:, 0] * (1.0 - x[:, 1])
            f3 = (1.0 + g) * (1.0 - x[:, 0])
        else:  # idx == 7
            f1 = (1.0 + g) * np.cos(np.pi / 2.0 * x[:, 0]) * np.cos(np.pi / 2.0 * x[:, 1])
            f2 = (1.0 + g) * np.cos(np.pi / 2.0 * x[:, 0]) * np.sin(np.pi / 2.0 * x[:, 1])
            f3 = (1.0 + g) * np.sin(np.pi / 2.0 * x[:, 0])
        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, n_pareto_points=200):
        idx = self._IDX
        n = max(2, int(n_pareto_points))

        if idx in {1, 5}:
            x = np.linspace(0.0, 1.0, n)
            return np.column_stack([x, 1.0 - np.sqrt(x)])

        if idx == 2:
            x = np.linspace(0.0, 1.0, n)
            return np.column_stack([x, 1.0 - x**2])

        if idx == 3:
            x = np.linspace(0.0, 1.0, n)
            return np.column_stack([x, np.sqrt(1.0 - x**2)])

        if idx == 4:
            x = np.linspace(0.0, 1.0, n)
            pf = np.column_stack([x, 1.0 - np.sqrt(x) * np.cos(2.0 * np.pi * x) ** 2])
            return pf[_nondominated_mask(pf)]

        if idx == 6:
            return _uniform_simplex(n, 3)

        r = _uniform_simplex(n, 3)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)


class MOEADM2M_F1(_BaseMOEADM2M):
    _IDX = 1


class MOEADM2M_F2(_BaseMOEADM2M):
    _IDX = 2


class MOEADM2M_F3(_BaseMOEADM2M):
    _IDX = 3


class MOEADM2M_F4(_BaseMOEADM2M):
    _IDX = 4


class MOEADM2M_F5(_BaseMOEADM2M):
    _IDX = 5


class MOEADM2M_F6(_BaseMOEADM2M):
    _IDX = 6


class MOEADM2M_F7(_BaseMOEADM2M):
    _IDX = 7


_CPU = [f"MOEADM2M_F{i}" for i in range(1, 8)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
