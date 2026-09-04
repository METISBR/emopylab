from __future__ import annotations

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


def _clip_x(x, xl, xu, xp):
    x = xp.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[xp.newaxis, :]
    return xp.clip(x, xp.asarray(xl, dtype=float), xp.asarray(xu, dtype=float))


def _to_numpy(x):
    return np.asarray(x, dtype=float)


class _BaseZDT(Problem):
    _USE_JAX = False

    def _xp(self):
        return _xp(self._USE_JAX)


class ZDT1(_BaseZDT):
    """Ref: Zitzler, Deb, and Thiele, 2000."""

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 30
        kwargs.pop("n_obj", None)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        f1 = x[:, 0]
        g = 1.0 + 9.0 * xp.mean(x[:, 1:], axis=1)
        f2 = g * (1.0 - xp.sqrt(f1 / g))
        out["F"] = _to_numpy(xp.column_stack([f1, f2]))

    def _calc_pareto_front(self, n_pareto_points: int = 200):
        x = np.linspace(0.0, 1.0, int(max(2, n_pareto_points)))
        return np.column_stack([x, 1.0 - np.sqrt(x)])


class ZDT2(_BaseZDT):
    """Ref: Zitzler, Deb, and Thiele, 2000."""

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 30
        kwargs.pop("n_obj", None)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        f1 = x[:, 0]
        g = 1.0 + 9.0 * xp.mean(x[:, 1:], axis=1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        out["F"] = _to_numpy(xp.column_stack([f1, f2]))

    def _calc_pareto_front(self, n_pareto_points: int = 200):
        x = np.linspace(0.0, 1.0, int(max(2, n_pareto_points)))
        return np.column_stack([x, 1.0 - x**2])


class ZDT3(_BaseZDT):
    """Ref: Zitzler, Deb, and Thiele, 2000."""

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 30
        kwargs.pop("n_obj", None)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        f1 = x[:, 0]
        g = 1.0 + 9.0 * xp.mean(x[:, 1:], axis=1)
        f2 = g * (1.0 - xp.sqrt(f1 / g) - (f1 / g) * xp.sin(10.0 * xp.pi * f1))
        out["F"] = _to_numpy(xp.column_stack([f1, f2]))

    def _calc_pareto_front(self, n_pareto_points: int = 200):
        regions = [
            (0.0, 0.0830015349),
            (0.1822287800, 0.2577623634),
            (0.4093136748, 0.4538821041),
            (0.6183967944, 0.6525117038),
            (0.8233317983, 0.8518328654),
        ]
        n_total = int(max(10, n_pareto_points))
        per = max(2, n_total // len(regions))
        parts = []
        for lo, hi in regions:
            x = np.linspace(lo, hi, per)
            y = 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)
            parts.append(np.column_stack([x, y]))
        return np.vstack(parts)

class ZDT4(_BaseZDT):
    """Ref: Zitzler, Deb, and Thiele, 2000."""

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        xl = np.full(n_var, -5.0, dtype=float)
        xu = np.full(n_var, 5.0, dtype=float)
        xl[0] = 0.0
        xu[0] = 1.0
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        f1 = x[:, 0]
        g = 1.0 + 10.0 * (self.n_var - 1) + xp.sum(x[:, 1:] ** 2 - 10.0 * xp.cos(4.0 * xp.pi * x[:, 1:]), axis=1)
        f2 = g * (1.0 - xp.sqrt(f1 / g))
        out["F"] = _to_numpy(xp.column_stack([f1, f2]))

    def _calc_pareto_front(self, n_pareto_points: int = 200):
        x = np.linspace(0.0, 1.0, int(max(2, n_pareto_points)))
        return np.column_stack([x, 1.0 - np.sqrt(x)])


class ZDT5(_BaseZDT):
    """Ref: Zitzler, Deb, and Thiele, 2000. Binary."""

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 80
        kwargs.pop("n_obj", None)
        n_var = int(np.ceil(max(n_var - 30, 1) / 5.0) * 5 + 30)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = xp.asarray(x)
        if x.ndim == 1:
            x = x[xp.newaxis, :]
        x = (x > 0.5).astype(float)

        n = x.shape[0]
        blocks = 1 + (self.n_var - 30) // 5
        u = xp.zeros((n, blocks), dtype=float)
        u = u.at[:, 0].set(xp.sum(x[:, :30], axis=1)) if xp is not np else u
        if xp is np:
            u[:, 0] = np.sum(np.asarray(x)[:, :30], axis=1)

        for i in range(1, blocks):
            lo = 30 + (i - 1) * 5
            hi = lo + 5
            s = xp.sum(x[:, lo:hi], axis=1)
            if xp is np:
                u[:, i] = np.asarray(s)
            else:
                u = u.at[:, i].set(s)

        v = xp.zeros_like(u)
        v = xp.where(u < 5, 2.0 + u, v)
        v = xp.where(u == 5, 1.0, v)

        f1 = 1.0 + u[:, 0]
        g = xp.sum(v[:, 1:], axis=1)
        f2 = g / f1
        out["F"] = _to_numpy(xp.column_stack([f1, f2]))

    def _calc_pareto_front(self, n_pareto_points: int = 200):
        n_points = int(max(2, n_pareto_points))
        blocks = 1 + (self.n_var - 30) // 5
        f1 = np.linspace(1.0, 31.0, n_points)
        f2 = (blocks - 1) / f1
        return np.column_stack([f1, f2])


class ZDT6(_BaseZDT):
    """Ref: Zitzler, Deb, and Thiele, 2000."""

    def __init__(self, n_var: int | None = None, n_obj: int | None = None, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        f1 = 1.0 - xp.exp(-4.0 * x[:, 0]) * xp.sin(6.0 * xp.pi * x[:, 0]) ** 6
        g = 1.0 + 9.0 * xp.mean(x[:, 1:], axis=1) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        out["F"] = _to_numpy(xp.column_stack([f1, f2]))

    def _calc_pareto_front(self, n_pareto_points: int = 200):
        x = np.linspace(0.2807753191, 1.0, int(max(2, n_pareto_points)))
        return np.column_stack([x, 1.0 - x**2])


_CPU = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT5", "ZDT6"]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
