from __future__ import annotations

"""
VNT benchmark family.

Reference
---------
R. Viennet, C. Fonteix, and I. Marc.
Multicriteria optimization using a genetic algorithm for determining a Pareto set.
International Journal of Systems Science, 1996, 27(2): 255-260.
"""

import numpy as np
from core.problem import Problem

try:
    import jax.numpy as jnp
except Exception:  # noqa: BLE001
    jnp = None


def _xp(use_jax: bool):
    if use_jax and jnp is not None:
        return jnp
    return np


def _to_numpy(x):
    return np.asarray(x, dtype=float)


def _clip_x(x, xl, xu, xp):
    arr = xp.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[xp.newaxis, :]
    return xp.clip(arr, xp.asarray(xl, dtype=float), xp.asarray(xu, dtype=float))


def _nondominated_mask(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return keep


class _BaseVNT(Problem):
    _USE_JAX = False

    def _xp(self):
        return _xp(self._USE_JAX)

    def _sample_grid(self, n: int) -> np.ndarray:
        n = max(10, int(np.ceil(np.sqrt(max(100, int(n))))))
        x = np.linspace(float(self.xl[0]), float(self.xu[0]), n)
        y = np.linspace(float(self.xl[1]), float(self.xu[1]), n)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _calc_pareto_front(self, n_pareto_points=200):
        x = self._sample_grid(int(n_pareto_points))
        out = {}
        self._evaluate(x, out)
        f = np.asarray(out["F"], dtype=float)
        mask = _nondominated_mask(f)
        return f[mask]


class VNT1(_BaseVNT):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=3, xl=np.array([-2.0, -2.0]), xu=np.array([2.0, 2.0]), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        f1 = x[:, 0] ** 2 + (x[:, 1] - 1.0) ** 2
        f2 = x[:, 0] ** 2 + (x[:, 1] + 1.0) ** 2 + 1.0
        f3 = (x[:, 0] - 1.0) ** 2 + x[:, 1] ** 2 + 2.0
        out["F"] = _to_numpy(xp.column_stack([f1, f2, f3]))


class VNT2(_BaseVNT):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=3, xl=np.array([-4.0, -4.0]), xu=np.array([4.0, 4.0]), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        f1 = (x[:, 0] - 2.0) ** 2 / 2.0 + (x[:, 1] + 1.0) ** 2 / 13.0 + 3.0
        f2 = (x[:, 0] + x[:, 1] - 3.0) ** 2 / 36.0 + (-x[:, 0] + x[:, 1] + 2.0) ** 2 / 8.0 - 17.0
        f3 = (x[:, 0] + 2.0 * x[:, 1] - 1.0) ** 2 / 175.0 + (2.0 * x[:, 1] - x[:, 0]) ** 2 / 17.0 - 13.0
        out["F"] = _to_numpy(xp.column_stack([f1, f2, f3]))


class VNT3(_BaseVNT):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=3, xl=np.array([-3.0, -3.0]), xu=np.array([3.0, 3.0]), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        temp = x[:, 0] ** 2 + x[:, 1] ** 2
        f1 = 0.5 * temp + xp.sin(temp)
        f2 = (3.0 * x[:, 0] - 2.0 * x[:, 1] + 4.0) ** 2 / 8.0 + (x[:, 0] - x[:, 1] + 1.0) ** 2 / 27.0 + 15.0
        f3 = 1.0 / (temp + 1.0) - 1.1 * xp.exp(-temp)
        out["F"] = _to_numpy(xp.column_stack([f1, f2, f3]))


class VNT4(_BaseVNT):
    def __init__(self, **kwargs):
        super().__init__(
            n_var=2,
            n_obj=3,
            n_ieq_constr=3,
            xl=np.array([-4.0, -4.0]),
            xu=np.array([4.0, 4.0]),
            vtype=float,
            **kwargs,
        )

    def _sample_grid(self, n: int) -> np.ndarray:
        n = max(10, int(np.ceil(np.sqrt(max(100, int(n))))))
        x_vals = np.linspace(-1.0, 1.2, n)
        rows = []
        for xv in x_vals:
            y_vals = np.linspace(xv - 2.0, -4.0 * xv + 4.0, n)
            rows.append(np.column_stack([np.full(n, xv), y_vals]))
        return np.vstack(rows)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        f1 = (x[:, 0] - 2.0) ** 2 / 2.0 + (x[:, 1] + 1.0) ** 2 / 13.0 + 3.0
        f2 = (x[:, 0] + x[:, 1] - 3.0) ** 2 / 175.0 + (2.0 * x[:, 1] - x[:, 0]) ** 2 / 17.0 - 13.0
        f3 = (3.0 * x[:, 0] - 2.0 * x[:, 1] + 4.0) ** 2 / 8.0 + (x[:, 0] - x[:, 1] + 1.0) ** 2 / 27.0 + 15.0
        g1 = x[:, 1] + 4.0 * x[:, 0] - 4.0
        g2 = -1.0 - x[:, 0]
        g3 = x[:, 0] - 2.0 - x[:, 1]
        out["F"] = _to_numpy(xp.column_stack([f1, f2, f3]))
        out["G"] = _to_numpy(xp.column_stack([g1, g2, g3]))


_CPU = ["VNT1", "VNT2", "VNT3", "VNT4"]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
