from __future__ import annotations

"""
Multitasking multi-objective benchmark instances.

Reference
---------
A. Gupta, Y. Ong, L. Feng, and K. C. Tan.
Multiobjective multifactorial optimization in evolutionary multitasking.
IEEE Transactions on Cybernetics, 2017, 47(7): 1652-1665.
"""

import math

import numpy as np
from core.problem import Problem


def _build_rotation(dim: int, seed: int = 1) -> np.ndarray:
    if dim <= 0:
        return np.eye(0, dtype=float)
    rng = np.random.default_rng(seed)
    raw = rng.random((dim, dim))
    q, _ = np.linalg.qr(raw)
    return q


class _BaseInstance(Problem):
    def __init__(self, *, sub_d=(10, 10), constrained=False, task2_bound=512.0, **kwargs):
        self.sub_d = (int(sub_d[0]), int(sub_d[1]))
        self.sub_m = (2, 2)
        self.constrained = bool(constrained)

        n_var = max(self.sub_d) + 1
        self.l1 = np.array([0.0] + [-5.0] * (self.sub_d[0] - 1), dtype=float)
        self.u1 = np.array([1.0] + [5.0] * (self.sub_d[0] - 1), dtype=float)
        self.l2 = np.array([0.0] + [-float(task2_bound)] * (self.sub_d[1] - 1), dtype=float)
        self.u2 = np.array([1.0] + [float(task2_bound)] * (self.sub_d[1] - 1), dtype=float)

        rot_dim = max(0, n_var - 2)
        self.rotmx = _build_rotation(rot_dim, seed=1)

        xl = np.zeros(n_var, dtype=float)
        xu = np.ones(n_var, dtype=float)
        xl[-1] = 1.0
        xu[-1] = 2.0

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=1 if self.constrained else 0,
            xl=xl,
            xu=xu,
            vtype=float,
            **kwargs,
        )

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, max(2, int(n_pareto_points)))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _clip(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.clip(arr, self.xl, self.xu)
        arr[:, -1] = np.round(arr[:, -1])
        arr[:, -1] = np.clip(arr[:, -1], 1.0, 2.0)
        return arr

    def _task1_decoded(self, arr: np.ndarray) -> np.ndarray:
        x = self.l1 + arr[:, : self.sub_d[0]] * (self.u1 - self.l1)
        if x.shape[1] > 1:
            x[:, 1:] = x[:, 1:] @ self.rotmx.T
        return x

    def _task2_decoded(self, arr: np.ndarray) -> np.ndarray:
        x = self.l2 + arr[:, : self.sub_d[1]] * (self.u2 - self.l2)
        if x.shape[1] > 1:
            x[:, 1:] = x[:, 1:] @ self.rotmx.T
        return x


class Instance1(_BaseInstance):
    def __init__(self, sub_d=(10, 10), **kwargs):
        super().__init__(sub_d=sub_d, constrained=False, task2_bound=512.0, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        arr = self._clip(x)
        n = arr.shape[0]
        f = np.full((n, 2), np.nan, dtype=float)

        task = arr[:, -1].astype(int)
        mask1 = task == 1
        if np.any(mask1):
            x1 = self._task1_decoded(arr[mask1])
            g = 1.0 + 10.0 * (self.sub_d[0] - 1) + np.sum(x1[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x1[:, 1:]), axis=1)
            f[mask1, 0] = x1[:, 0]
            f[mask1, 1] = g * (1.0 - np.sqrt(x1[:, 0] / g))

        mask2 = task == 2
        if np.any(mask2):
            x2 = self._task2_decoded(arr[mask2])
            denom = np.sqrt(np.arange(2, self.sub_d[1] + 1, dtype=float))
            g = 2.0 + np.sum(x2[:, 1:] ** 2, axis=1) / 4000.0 - np.prod(np.cos(x2[:, 1:] / denom[None, :]), axis=1)
            f[mask2, 0] = x2[:, 0]
            f[mask2, 1] = g * (1.0 - np.sqrt(x2[:, 0] / g))

        out["F"] = f


class Instance2(_BaseInstance):
    def __init__(self, sub_d=(10, 10), **kwargs):
        super().__init__(sub_d=sub_d, constrained=True, task2_bound=32.0, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        arr = self._clip(x)
        n = arr.shape[0]
        f = np.full((n, 2), np.nan, dtype=float)
        g_con = np.zeros((n, 1), dtype=float)

        task = arr[:, -1].astype(int)
        mask1 = task == 1
        if np.any(mask1):
            x1 = self._task1_decoded(arr[mask1])
            g = 1.0 + 10.0 * (self.sub_d[0] - 1) + np.sum(x1[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x1[:, 1:]), axis=1)
            f1 = x1[:, 0]
            f2 = g * (1.0 - np.sqrt(f1 / g))
            f[mask1, 0] = f1
            f[mask1, 1] = f2

            theta = -0.05 * np.pi
            a = 40.0
            b = 5.0
            c = 1.0
            d = 6.0
            e = 0.0
            gc = (
                a * np.abs(np.sin(b * np.pi * (np.sin(theta) * (f2 - e) + np.cos(theta) * f1) ** c)) ** d
                - np.cos(theta) * (f2 - e)
                + np.sin(theta) * f1
            )
            g_con[mask1, 0] = gc

        mask2 = task == 2
        if np.any(mask2):
            x2 = self._task2_decoded(arr[mask2])
            sq = np.sum(x2[:, 1:] ** 2, axis=1)
            term1 = -20.0 * np.exp(-0.2 * np.sqrt(sq / (self.sub_d[1] - 1)))
            term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x2[:, 1:]), axis=1) / (self.sub_d[1] - 1))
            g = term1 + term2 + 21.0 + math.e
            f[mask2, 0] = x2[:, 0]
            f[mask2, 1] = g * (1.0 - np.sqrt(x2[:, 0] / g))

        out["F"] = f
        out["G"] = g_con


class Instance1_JAX(Instance1):
    pass


class Instance2_JAX(Instance2):
    pass


__all__ = ["Instance1", "Instance2", "Instance1_JAX", "Instance2_JAX"]
