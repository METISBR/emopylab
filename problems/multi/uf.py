from __future__ import annotations

"""
UF benchmark family (CEC 2009) converted for local EmoPyLab use.

Reference
---------
Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, and S. Tiwari.
Multiobjective optimization test instances for the CEC 2009 special
session and competition. University of Essex, Working Report CES-487, 2009.
"""

import numpy as np
from core.problem import Problem


class _BaseUF(Problem):
    _USE_JAX = False

    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)

    def _calc_pareto_front(self, n_pareto_points: int = 300):
        n = max(32, int(n_pareto_points))
        idx = int(getattr(self, "_IDX", 1))

        if int(getattr(self, "n_obj", 2)) == 3:
            if idx in {8, 10}:
                return _sphere_pf(n)
            if idx == 9:
                return _uf9_pf(n)
            return None

        x = np.linspace(0.0, 1.0, n, dtype=float)
        if idx in {1, 2, 3}:
            return np.column_stack([x, 1.0 - np.sqrt(x)])
        if idx == 4:
            return np.column_stack([x, 1.0 - x**2])
        if idx == 5:
            bump = 0.15 * np.abs(np.sin(20.0 * np.pi * x))
            return np.column_stack([x + bump, 1.0 - x + bump])
        if idx == 6:
            bump = np.maximum(0.0, 0.7 * np.sin(4.0 * np.pi * x))
            return np.column_stack([x + bump, 1.0 - x + bump])
        if idx == 7:
            t = x**0.2
            return np.column_stack([t, 1.0 - t])
        return None


def _sphere_pf(n: int) -> np.ndarray:
    r = np.random.default_rng(1).random((max(2, int(n)), 3))
    nrm = np.linalg.norm(r, axis=1, keepdims=True)
    nrm[nrm == 0.0] = 1.0
    return r / nrm


def _nondominated_front_3d(f: np.ndarray) -> np.ndarray:
    arr = np.asarray(f, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.empty((0, 3), dtype=float)
    keep = np.ones(arr.shape[0], dtype=bool)
    for i in range(arr.shape[0]):
        if not keep[i]:
            continue
        dominated = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
            continue
        keep &= ~(np.all(arr[i] <= arr, axis=1) & np.any(arr[i] < arr, axis=1))
        keep[i] = True
    out = arr[keep]
    if out.shape[0] <= 1:
        return out
    order = np.lexsort((out[:, 2], out[:, 1], out[:, 0]))
    return out[order]


def _uf9_pf(n: int) -> np.ndarray:
    # PF approximation in objective space by sampling the decision manifold (g=0).
    n = max(64, int(n))
    side = max(16, int(np.ceil(np.sqrt(n * 2))))
    x0 = np.linspace(0.0, 1.0, side, dtype=float)
    x1 = np.linspace(0.0, 1.0, side, dtype=float)
    xx0, xx1 = np.meshgrid(x0, x1, indexing="xy")
    x0f = xx0.ravel()
    x1f = xx1.ravel()
    term = np.maximum(0.0, 1.1 * (1.0 - 4.0 * (2.0 * x0f - 1.0) ** 2))
    f1 = 0.5 * (term + 2.0 * x0f) * x1f
    f2 = 0.5 * (term - 2.0 * x0f + 2.0) * x1f
    f3 = 1.0 - x1f
    pf = _nondominated_front_3d(np.column_stack([f1, f2, f3]))
    if pf.shape[0] > n:
        idx = np.linspace(0, pf.shape[0] - 1, n, dtype=int)
        pf = pf[idx]
    return pf


class UF1(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(n_var - 1)))
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(2, self.n_var, 2)
        j2 = np.arange(1, self.n_var, 2)
        idx = np.arange(1, self.n_var + 1)
        y = x - np.sin(6 * np.pi * x[:, [0]] + idx[None, :] * np.pi / self.n_var)
        f1 = x[:, 0] + 2 * np.mean(y[:, j1] ** 2, axis=1)
        f2 = 1 - np.sqrt(x[:, 0]) + 2 * np.mean(y[:, j2] ** 2, axis=1)
        out["F"] = np.column_stack([f1, f2])


class UF2(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(n_var - 1)))
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(2, self.n_var, 2)
        j2 = np.arange(1, self.n_var, 2)
        y = np.zeros_like(x)

        x1a = np.tile(x[:, [0]], (1, len(j1)))
        jj1 = (j1 + 1)[None, :]
        y[:, j1] = x[:, j1] - (0.3 * x1a**2 * np.cos(24 * np.pi * x1a + 4 * jj1 * np.pi / self.n_var) + 0.6 * x1a) * np.cos(
            6 * np.pi * x1a + jj1 * np.pi / self.n_var
        )

        x1b = np.tile(x[:, [0]], (1, len(j2)))
        jj2 = (j2 + 1)[None, :]
        y[:, j2] = x[:, j2] - (0.3 * x1b**2 * np.cos(24 * np.pi * x1b + 4 * jj2 * np.pi / self.n_var) + 0.6 * x1b) * np.sin(
            6 * np.pi * x1b + jj2 * np.pi / self.n_var
        )

        f1 = x[:, 0] + 2 * np.mean(y[:, j1] ** 2, axis=1)
        f2 = 1 - np.sqrt(x[:, 0]) + 2 * np.mean(y[:, j2] ** 2, axis=1)
        out["F"] = np.column_stack([f1, f2])


class UF3(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(2, self.n_var, 2)
        j2 = np.arange(1, self.n_var, 2)
        idx = np.arange(1, self.n_var + 1)
        y = x - x[:, [0]] ** (0.5 * (1 + 3 * (idx[None, :] - 2) / (self.n_var - 2)))

        t1 = 4 * np.sum(y[:, j1] ** 2, axis=1) - 2 * np.prod(np.cos(20 * y[:, j1] * np.pi / np.sqrt((j1 + 1)[None, :])), axis=1) + 2
        t2 = 4 * np.sum(y[:, j2] ** 2, axis=1) - 2 * np.prod(np.cos(20 * y[:, j2] * np.pi / np.sqrt((j2 + 1)[None, :])), axis=1) + 2
        f1 = x[:, 0] + 2 / len(j1) * t1
        f2 = 1 - np.sqrt(x[:, 0]) + 2 / len(j2) * t2
        out["F"] = np.column_stack([f1, f2])


class UF4(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0], -2 * np.ones(n_var - 1)))
        xu = np.concatenate(([1.0], 2 * np.ones(n_var - 1)))
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(2, self.n_var, 2)
        j2 = np.arange(1, self.n_var, 2)
        idx = np.arange(1, self.n_var + 1)
        y = x - np.sin(6 * np.pi * x[:, [0]] + idx[None, :] * np.pi / self.n_var)
        hy = np.abs(y) / (1 + np.exp(2 * np.abs(y)))
        f1 = x[:, 0] + 2 * np.mean(hy[:, j1], axis=1)
        f2 = 1 - x[:, 0] ** 2 + 2 * np.mean(hy[:, j2], axis=1)
        out["F"] = np.column_stack([f1, f2])


class UF5(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(n_var - 1)))
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(2, self.n_var, 2)
        j2 = np.arange(1, self.n_var, 2)
        idx = np.arange(1, self.n_var + 1)
        y = x - np.sin(6 * np.pi * x[:, [0]] + idx[None, :] * np.pi / self.n_var)
        hy = 2 * y**2 - np.cos(4 * np.pi * y) + 1
        bump = (1 / 20 + 0.1) * np.abs(np.sin(20 * np.pi * x[:, 0]))
        f1 = x[:, 0] + bump + 2 * np.mean(hy[:, j1], axis=1)
        f2 = 1 - x[:, 0] + bump + 2 * np.mean(hy[:, j2], axis=1)
        out["F"] = np.column_stack([f1, f2])


class UF6(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(n_var - 1)))
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(2, self.n_var, 2)
        j2 = np.arange(1, self.n_var, 2)
        idx = np.arange(1, self.n_var + 1)
        y = x - np.sin(6 * np.pi * x[:, [0]] + idx[None, :] * np.pi / self.n_var)

        t1 = 4 * np.sum(y[:, j1] ** 2, axis=1) - 2 * np.prod(np.cos(20 * y[:, j1] * np.pi / np.sqrt((j1 + 1)[None, :])), axis=1) + 2
        t2 = 4 * np.sum(y[:, j2] ** 2, axis=1) - 2 * np.prod(np.cos(20 * y[:, j2] * np.pi / np.sqrt((j2 + 1)[None, :])), axis=1) + 2
        bump = np.maximum(0.0, 2 * (1 / 4 + 0.1) * np.sin(4 * np.pi * x[:, 0]))
        f1 = x[:, 0] + bump + 2 / len(j1) * t1
        f2 = 1 - x[:, 0] + bump + 2 / len(j2) * t2
        out["F"] = np.column_stack([f1, f2])


class UF7(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(n_var - 1)))
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(2, self.n_var, 2)
        j2 = np.arange(1, self.n_var, 2)
        idx = np.arange(1, self.n_var + 1)
        y = x - np.sin(6 * np.pi * x[:, [0]] + idx[None, :] * np.pi / self.n_var)
        f1 = x[:, 0] ** 0.2 + 2 * np.mean(y[:, j1] ** 2, axis=1)
        f2 = 1 - x[:, 0] ** 0.2 + 2 * np.mean(y[:, j2] ** 2, axis=1)
        out["F"] = np.column_stack([f1, f2])


class UF8(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0, 0.0], -2 * np.ones(n_var - 2)))
        xu = np.concatenate(([1.0, 1.0], 2 * np.ones(n_var - 2)))
        super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(3, self.n_var, 3)
        j2 = np.arange(4, self.n_var, 3)
        j3 = np.arange(2, self.n_var, 3)
        idx = np.arange(1, self.n_var + 1)
        y = x - 2 * x[:, [1]] * np.sin(2 * np.pi * x[:, [0]] + idx[None, :] * np.pi / self.n_var)

        f1 = np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, j1] ** 2, axis=1)
        f2 = np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, j2] ** 2, axis=1)
        f3 = np.sin(0.5 * x[:, 0] * np.pi) + 2 * np.mean(y[:, j3] ** 2, axis=1)
        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, n_pareto_points: int = 300):
        return _sphere_pf(n_pareto_points)


class UF9(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0, 0.0], -2 * np.ones(n_var - 2)))
        xu = np.concatenate(([1.0, 1.0], 2 * np.ones(n_var - 2)))
        super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(3, self.n_var, 3)
        j2 = np.arange(4, self.n_var, 3)
        j3 = np.arange(2, self.n_var, 3)
        idx = np.arange(1, self.n_var + 1)
        y = x - 2 * x[:, [1]] * np.sin(2 * np.pi * x[:, [0]] + idx[None, :] * np.pi / self.n_var)

        term = np.maximum(0.0, 1.1 * (1 - 4 * (2 * x[:, 0] - 1) ** 2))
        f1 = 0.5 * (term + 2 * x[:, 0]) * x[:, 1] + 2 * np.mean(y[:, j1] ** 2, axis=1)
        f2 = 0.5 * (term - 2 * x[:, 0] + 2) * x[:, 1] + 2 * np.mean(y[:, j2] ** 2, axis=1)
        f3 = 1 - x[:, 1] + 2 * np.mean(y[:, j3] ** 2, axis=1)
        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, n_pareto_points: int = 300):
        return _uf9_pf(n_pareto_points)


class UF10(_BaseUF):
    def __init__(self, n_var: int = 30, **kwargs):
        n_var = int(n_var)
        xl = np.concatenate(([0.0, 0.0], -2 * np.ones(n_var - 2)))
        xu = np.concatenate(([1.0, 1.0], 2 * np.ones(n_var - 2)))
        super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        j1 = np.arange(3, self.n_var, 3)
        j2 = np.arange(4, self.n_var, 3)
        j3 = np.arange(2, self.n_var, 3)
        idx = np.arange(1, self.n_var + 1)
        y = x - 2 * x[:, [1]] * np.sin(2 * np.pi * x[:, [0]] + idx[None, :] * np.pi / self.n_var)
        y = 4 * y**2 - np.cos(8 * np.pi * y) + 1

        f1 = np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, j1], axis=1)
        f2 = np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, j2], axis=1)
        f3 = np.sin(0.5 * x[:, 0] * np.pi) + 2 * np.mean(y[:, j3], axis=1)
        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, n_pareto_points: int = 300):
        return _sphere_pf(n_pareto_points)


_CPU = [f"UF{i}" for i in range(1, 11)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
