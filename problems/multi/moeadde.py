from __future__ import annotations

"""
MOEADDE benchmark family.

Reference
---------
H. Li and Q. Zhang.
Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II.
IEEE Transactions on Evolutionary Computation, 2009, 13(2): 284-302.
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


class _BaseMOEADDE(Problem):
    _IDX = 1

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        idx = self._IDX
        m_default = 3 if idx == 6 else 2
        d_default = 10 if idx in {6, 7, 8} else 30
        m = m_default if n_obj is None else int(n_obj)
        d = d_default if n_var is None else int(n_var)
        d = max(m, d)
        super().__init__(n_var=d, n_obj=m, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.clip(_as_2d(x), self.xl, self.xu)
        d = self.n_var
        idx = self._IDX

        if idx in {1, 2, 3, 4, 5, 7, 8, 9}:
            j1 = np.arange(2, d, 2, dtype=int)  # MATLAB 3:2:D
            j2 = np.arange(1, d, 2, dtype=int)  # MATLAB 2:2:D
            x1 = x[:, [0]]

            if idx == 1:
                exp1 = ((1.0 + 3.0 * ((j1 + 1) - 2) / (d - 2)) / 2.0)[None, :]
                exp2 = ((1.0 + 3.0 * ((j2 + 1) - 2) / (d - 2)) / 2.0)[None, :]
                f1 = x[:, 0] + 2.0 * np.mean((x[:, j1] - x1**exp1) ** 2, axis=1)
                f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 * np.mean((x[:, j2] - x1**exp2) ** 2, axis=1)

            elif idx == 2:
                t1 = np.sin(6.0 * np.pi * x1 + ((j1 + 1) * np.pi / d)[None, :])
                t2 = np.sin(6.0 * np.pi * x1 + ((j2 + 1) * np.pi / d)[None, :])
                f1 = x[:, 0] + 2.0 * np.mean((x[:, j1] - t1) ** 2, axis=1)
                f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 * np.mean((x[:, j2] - t2) ** 2, axis=1)

            elif idx == 3:
                c1 = 0.8 * x1 * np.cos(6.0 * np.pi * x1 + ((j1 + 1) * np.pi / d)[None, :])
                c2 = 0.8 * x1 * np.sin(6.0 * np.pi * x1 + ((j2 + 1) * np.pi / d)[None, :])
                f1 = x[:, 0] + 2.0 * np.mean((x[:, j1] - c1) ** 2, axis=1)
                f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 * np.mean((x[:, j2] - c2) ** 2, axis=1)

            elif idx == 4:
                c1 = 0.8 * x1 * np.cos(2.0 * np.pi * x1 + ((j1 + 1) * np.pi / d / 3.0)[None, :])
                c2 = 0.8 * x1 * np.sin(6.0 * np.pi * x1 + ((j2 + 1) * np.pi / d)[None, :])
                f1 = x[:, 0] + 2.0 * np.mean((x[:, j1] - c1) ** 2, axis=1)
                f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 * np.mean((x[:, j2] - c2) ** 2, axis=1)

            elif idx == 5:
                x1j1 = np.tile(x1, (1, len(j1)))
                x1j2 = np.tile(x1, (1, len(j2)))
                a1 = 0.3 * x1j1**2 * np.cos(24.0 * np.pi * x1j1 + (4.0 * (j1 + 1) * np.pi / d)[None, :]) + 0.6 * x1j1
                a2 = 0.3 * x1j2**2 * np.cos(24.0 * np.pi * x1j2 + (4.0 * (j2 + 1) * np.pi / d)[None, :]) + 0.6 * x1j2
                c1 = a1 * np.cos(6.0 * np.pi * x1j1 + ((j1 + 1) * np.pi / d)[None, :])
                c2 = a2 * np.sin(6.0 * np.pi * x1j2 + ((j2 + 1) * np.pi / d)[None, :])
                f1 = x[:, 0] + 2.0 * np.mean((x[:, j1] - c1) ** 2, axis=1)
                f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 * np.mean((x[:, j2] - c2) ** 2, axis=1)

            elif idx == 7:
                j_all = np.arange(1, d + 1, dtype=float)
                exp_all = ((1.0 + 3.0 * (j_all - 2.0) / (d - 2.0)) / 2.0)[None, :]
                y = x - x1**exp_all
                f1 = x[:, 0] + 2.0 * np.mean(4.0 * y[:, j1] ** 2 - np.cos(8.0 * np.pi * y[:, j1]) + 1.0, axis=1)
                f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 * np.mean(4.0 * y[:, j2] ** 2 - np.cos(8.0 * np.pi * y[:, j2]) + 1.0, axis=1)

            elif idx == 8:
                j_all = np.arange(1, d + 1, dtype=float)
                exp_all = ((1.0 + 3.0 * (j_all - 2.0) / (d - 2.0)) / 2.0)[None, :]
                y = x - x1**exp_all
                term1 = 4.0 * np.sum(y[:, j1] ** 2, axis=1) - 2.0 * np.prod(
                    np.cos(20.0 * np.pi * y[:, j1] / np.sqrt((j1 + 1).astype(float))[None, :]), axis=1
                ) + 2.0
                term2 = 4.0 * np.sum(y[:, j2] ** 2, axis=1) - 2.0 * np.prod(
                    np.cos(20.0 * np.pi * y[:, j2] / np.sqrt((j2 + 1).astype(float))[None, :]), axis=1
                ) + 2.0
                f1 = x[:, 0] + 2.0 / max(1, len(j1)) * term1
                f2 = 1.0 - np.sqrt(x[:, 0]) + 2.0 / max(1, len(j2)) * term2

            else:  # idx == 9
                t1 = np.sin(6.0 * np.pi * x1 + ((j1 + 1) * np.pi / d)[None, :])
                t2 = np.sin(6.0 * np.pi * x1 + ((j2 + 1) * np.pi / d)[None, :])
                f1 = x[:, 0] + 2.0 * np.mean((x[:, j1] - t1) ** 2, axis=1)
                f2 = 1.0 - x[:, 0] ** 2 + 2.0 * np.mean((x[:, j2] - t2) ** 2, axis=1)

            out["F"] = np.column_stack([f1, f2])
            return

        if idx == 6:
            j1 = np.arange(3, d, 3, dtype=int)  # 4:3:D
            j2 = np.arange(4, d, 3, dtype=int)  # 5:3:D
            j3 = np.arange(2, d, 3, dtype=int)  # 3:3:D
            x1 = x[:, [0]]
            x2 = x[:, [1]]

            t1 = 2.0 * x2 * np.sin(2.0 * np.pi * x1 + ((j1 + 1) * np.pi / d)[None, :])
            t2 = 2.0 * x2 * np.sin(2.0 * np.pi * x1 + ((j2 + 1) * np.pi / d)[None, :])
            t3 = 2.0 * x2 * np.sin(2.0 * np.pi * x1 + ((j3 + 1) * np.pi / d)[None, :])

            f1 = np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1]) + 2.0 * np.mean((x[:, j1] - t1) ** 2, axis=1)
            f2 = np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1]) + 2.0 * np.mean((x[:, j2] - t2) ** 2, axis=1)
            f3 = np.sin(0.5 * np.pi * x[:, 0]) + 2.0 * np.mean((x[:, j3] - t3) ** 2, axis=1)
            out["F"] = np.column_stack([f1, f2, f3])
            return

        raise RuntimeError(f"Unsupported MOEADDE index: {idx}")

    def _calc_pareto_front(self, n_pareto_points=200):
        idx = self._IDX
        n = max(2, int(n_pareto_points))
        if self.n_obj == 3:
            r = _uniform_simplex(n, 3)
            return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

        x = np.linspace(0.0, 1.0, n)
        if idx == 9:
            return np.column_stack([x, 1.0 - x**2])
        return np.column_stack([x, 1.0 - np.sqrt(x)])


class MOEADDE_F1(_BaseMOEADDE):
    _IDX = 1


class MOEADDE_F2(_BaseMOEADDE):
    _IDX = 2


class MOEADDE_F3(_BaseMOEADDE):
    _IDX = 3


class MOEADDE_F4(_BaseMOEADDE):
    _IDX = 4


class MOEADDE_F5(_BaseMOEADDE):
    _IDX = 5


class MOEADDE_F6(_BaseMOEADDE):
    _IDX = 6


class MOEADDE_F7(_BaseMOEADDE):
    _IDX = 7


class MOEADDE_F8(_BaseMOEADDE):
    _IDX = 8


class MOEADDE_F9(_BaseMOEADDE):
    _IDX = 9


_CPU = [f"MOEADDE_F{i}" for i in range(1, 10)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
