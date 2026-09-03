from __future__ import annotations

"""
LSMMOP multitasking sparse multi-objective benchmark family.

Reference
---------
C. Wu, Y. Tian, L. Zhang, X. Xiang, and X. Zhang.
A sparsity knowledge transfer-based evolutionary algorithm for large-scale
multitasking multi-objective optimization.
IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2582-2595.
"""

import numpy as np

from core.problem import Problem


def _as_2d(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _g1(x, t):
    return (x - t) ** 2


def _g2(x, t):
    return 2.0 * (x - t) ** 2 + np.sin(2.0 * np.pi * (x - t)) ** 2


def _g3(x, t):
    expo = np.minimum(100.0 * (x - t) ** 2, 700.0)
    return 4.0 - (x - t) - 4.0 / np.exp(expo)


def _g4(x, t):
    return (x - np.pi / 3.0) ** 2 + t * np.sin(6.0 * np.pi * (x - np.pi / 3.0)) ** 2


def _shape_linear(x, m):
    n = x.shape[0]
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), x[:, : m - 1]]), axis=1))
    b = np.column_stack([np.ones((n, 1)), 1.0 - x[:, m - 2 :: -1]])
    return a * b


def _shape_sphere(x, m):
    n = x.shape[0]
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
    b = np.column_stack([np.ones((n, 1)), np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


def _shape_inverted(x, m):
    n = x.shape[0]
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), 1.0 - np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
    b = np.column_stack([np.ones((n, 1)), 1.0 - np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


class _BaseLSMMOP(Problem):
    _TASK_VARIANTS = (("g11", "linear"), ("g11", "linear"))

    def __init__(self, sub_d=(1000, 1000), sub_theta=(0.1, 0.1), **kwargs):
        self.sub_m = (2, 2)
        self.sub_d = [int(sub_d[0]), int(sub_d[1])]
        self.sub_theta = [float(sub_theta[0]), float(sub_theta[1])]

        self.n_tasks = len(self.sub_d)
        n_obj = max(self.sub_m)
        n_var = max(self.sub_d) + 1

        self.L = []
        self.U = []
        for d in self.sub_d:
            self.L.append(np.concatenate([np.zeros(n_obj - 1), -np.ones(d - n_obj + 1)]))
            self.U.append(np.concatenate([np.ones(n_obj - 1), 2.0 * np.ones(d - n_obj + 1)]))

        xl = np.concatenate([np.zeros(n_var - 1), np.array([1.0])])
        xu = np.concatenate([np.ones(n_var - 1), np.array([float(self.n_tasks)])])
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _decode(self, x: np.ndarray, task_idx: int) -> np.ndarray:
        d = self.sub_d[task_idx]
        xx = x[:, :d]
        l = self.L[task_idx][None, :]
        u = self.U[task_idx][None, :]
        return (l + (u - l) * xx) * (xx != 0.0)

    def _shape(self, x: np.ndarray, kind: str) -> np.ndarray:
        if kind == "linear":
            return _shape_linear(x, self.n_obj)
        if kind == "sphere":
            return _shape_sphere(x, self.n_obj)
        if kind == "inv":
            return _shape_inverted(x, self.n_obj)
        raise RuntimeError(f"Unknown shape: {kind}")

    def _g_variant(self, x: np.ndarray, theta: float, variant: str) -> np.ndarray:
        d = x.shape[1]
        m = self.n_obj
        tail = d - m + 1
        k = int(np.ceil(theta * tail))
        k = int(np.clip(k, 0, tail))
        st = m - 1

        if variant == "g11":
            a = x[:, st : st + k]
            b = x[:, st + k :]
            return np.sum(_g1(a, np.pi / 3.0), axis=1) + np.sum(_g2(b, 0.0), axis=1)

        if variant == "g22":
            a = x[:, st : st + k]
            b = x[:, st + k :]
            return np.sum(_g2(a, np.pi / 3.0), axis=1) + np.sum(_g3(b, 0.0), axis=1)

        if variant == "g33":
            g = np.sum(_g1(x[:, st : st + k], np.pi / 3.0), axis=1)
            rem = tail - k
            loops = int(np.ceil(rem / 10.0)) if rem > 0 else 0
            for i in range(loops):
                lo = st + k + i * 10
                hi = min(st + k + (i + 1) * 10, d)
                temp = 50.0 - np.sum(_g1(x[:, lo:hi], 0.0), axis=1)
                mask = temp < 50.0
                g[mask] += temp[mask]
            return g

        if variant == "g44":
            vals = np.sort(_g3(x[:, st:], 0.0), axis=1)
            keep = max(0, d - m - k + 1)
            return np.sum(vals[:, :keep], axis=1)

        if variant == "g55":
            t = x[:, st:]
            return np.sum(_g1(t, np.pi / 3.0) * _g2(t, 0.0), axis=1) + np.abs(k - np.sum(t != 0.0, axis=1))

        if variant == "g66":
            t = x[:, st:]
            tt = np.linspace(0.0, 1.0, tail, dtype=float)[None, :]
            vals = _g4(t, np.tile(tt, (x.shape[0], 1)))
            rank = np.argsort(vals, axis=1)
            vals_sorted = np.take_along_axis(vals, rank, axis=1)
            x_ranked = np.take_along_axis(t, rank, axis=1)
            mask = x_ranked == 0.0
            mask[:, :k] = False
            vals_sorted[mask] = 0.0
            return np.sum(vals_sorted, axis=1)

        if variant == "g77":
            a = x[:, st : st + k]
            b = x[:, st + k :]
            if b.shape[1] > 0:
                t = np.hstack([x[:, st + k + 1 :], x[:, st + k : st + k + 1]]) * 0.9
            else:
                t = b
            return np.sum(_g2(a, np.pi / 3.0), axis=1) + np.sum(_g2(b, t), axis=1)

        if variant == "g88":
            a = x[:, st : st + k]
            ta = np.mod(x[:, st + 1 : st + k + 1] + np.pi, 2.0) if k > 0 else a
            part1 = np.sum(_g3(a, ta), axis=1) if k > 0 else np.zeros(x.shape[0], dtype=float)

            b = x[:, st + k : d - 1]
            tb = x[:, st + k + 1 :] * 0.9
            part2 = np.sum(_g3(b, tb), axis=1) if b.shape[1] > 0 else np.zeros(x.shape[0], dtype=float)
            return part1 + part2

        raise RuntimeError(f"Unknown variant: {variant}")

    def _evaluate(self, x, out, *args, **kwargs):
        x = _as_2d(x)
        x = np.clip(x, self.xl, self.xu)
        x[:, :-1] = np.clip(x[:, :-1], 0.0, 1.0)
        labels = np.clip(np.rint(x[:, -1]).astype(int), 1, self.n_tasks)

        f = np.full((x.shape[0], self.n_obj), np.nan, dtype=float)
        for task in range(1, self.n_tasks + 1):
            mask = labels == task
            if not np.any(mask):
                continue

            x_task = self._decode(x[mask], task - 1)
            d = self.sub_d[task - 1]
            theta = self.sub_theta[task - 1]
            g_variant, shape_kind = self._TASK_VARIANTS[task - 1]
            g = self._g_variant(x_task, theta, g_variant)
            shape = self._shape(x_task, shape_kind)
            f[mask] = (1.0 + g / (d - self.n_obj + 1))[:, None] * shape

        out["F"] = f


class MMOP_NS1(_BaseLSMMOP):
    _TASK_VARIANTS = (("g33", "linear"), ("g77", "sphere"))


class MMOP_NS2(_BaseLSMMOP):
    _TASK_VARIANTS = (("g11", "linear"), ("g33", "linear"))


class MMOP_MS1(_BaseLSMMOP):
    _TASK_VARIANTS = (("g11", "linear"), ("g55", "inv"))


class MMOP_MS2(_BaseLSMMOP):
    _TASK_VARIANTS = (("g44", "inv"), ("g55", "inv"))


class MMOP_HS1(_BaseLSMMOP):
    _TASK_VARIANTS = (("g22", "linear"), ("g44", "inv"))


class MMOP_HS2(_BaseLSMMOP):
    _TASK_VARIANTS = (("g55", "inv"), ("g66", "inv"))


class MMOP_LS1(_BaseLSMMOP):
    _TASK_VARIANTS = (("g66", "inv"), ("g88", "sphere"))


class MMOP_LS2(_BaseLSMMOP):
    _TASK_VARIANTS = (("g33", "linear"), ("g88", "sphere"))


_CPU = [
    "MMOP_NS1",
    "MMOP_NS2",
    "MMOP_MS1",
    "MMOP_MS2",
    "MMOP_HS1",
    "MMOP_HS2",
    "MMOP_LS1",
    "MMOP_LS2",
]

for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
