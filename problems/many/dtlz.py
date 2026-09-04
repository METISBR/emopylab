from __future__ import annotations

import numpy as np
from core.problem import Problem

try:
    from util.ref_dirs import UniformReferenceDirectionFactory
except Exception:  # noqa: BLE001
    UniformReferenceDirectionFactory = None

try:
    import jax.numpy as jnp
except Exception:  # noqa: BLE001
    jnp = None


def _xp(use_jax: bool):
    if use_jax and jnp is not None:
        return jnp
    return np


def _to_numpy(arr):
    return np.asarray(arr, dtype=float)


def _clip_x(x, xl, xu, xp):
    x = xp.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[xp.newaxis, :]
    return xp.clip(x, xp.asarray(xl, dtype=float), xp.asarray(xu, dtype=float))


def _assign(arr, key, value, xp):
    if xp is np:
        arr[key] = value
        return arr
    return arr.at[key].set(value)


def _obj_dtlz1(x, m: int, g, xp):
    n = x.shape[0]
    left = xp.fliplr(xp.cumprod(xp.concatenate([xp.ones((n, 1)), x[:, : m - 1]], axis=1), axis=1))
    right = xp.concatenate([xp.ones((n, 1)), 1.0 - x[:, m - 2 :: -1]], axis=1)
    return 0.5 * xp.tile((1.0 + g)[:, None], (1, m)) * left * right


def _obj_dtlz2_like(x, m: int, g, xp):
    n = x.shape[0]
    left = xp.fliplr(
        xp.cumprod(
            xp.concatenate([xp.ones((n, 1)), xp.cos(x[:, : m - 1] * xp.pi / 2.0)], axis=1),
            axis=1,
        )
    )
    right = xp.concatenate([xp.ones((n, 1)), xp.sin(x[:, m - 2 :: -1] * xp.pi / 2.0)], axis=1)
    return xp.tile((1.0 + g)[:, None], (1, m)) * left * right


def _ref_dirs_default(n_obj: int, n_points: int = 200) -> np.ndarray:
    n_obj = max(2, int(n_obj))
    n_points = max(16, int(n_points))
    if UniformReferenceDirectionFactory is not None:
        try:
            if n_obj == 2:
                return np.asarray(UniformReferenceDirectionFactory(2, n_points=n_points).do(), dtype=float)
            # Best-effort partition search close to target size.
            best_dirs = None
            best_gap = None
            for p in range(1, 64):
                dirs = np.asarray(UniformReferenceDirectionFactory(n_obj, n_partitions=p).do(), dtype=float)
                gap = abs(int(dirs.shape[0]) - n_points)
                if best_dirs is None or gap < best_gap:
                    best_dirs, best_gap = dirs, gap
                if gap == 0:
                    break
                # Defensive stop for high-dimensional cases: avoid combinatorial blow-up when the count
                # is already far above the target and a reasonable candidate has been found.
                if best_dirs is not None and dirs.shape[0] > max(n_points * 4, n_points + 500):
                    break
            if best_dirs is not None and best_dirs.size:
                return best_dirs
        except Exception:  # noqa: BLE001
            pass
    # Fallback: deterministic simplex samples.
    rng = np.random.default_rng(1)
    return rng.dirichlet(np.ones(n_obj, dtype=float), size=n_points)


def _ensure_ref_dirs(ref_dirs, n_obj: int, n_points: int = 200) -> np.ndarray:
    if ref_dirs is None:
        return _ref_dirs_default(n_obj, n_points=n_points)
    arr = np.asarray(ref_dirs, dtype=float)
    if arr.ndim != 2 or arr.size == 0 or arr.shape[1] != int(n_obj):
        return _ref_dirs_default(n_obj, n_points=n_points)
    return arr


def _pf_simplex_scaled(ref_dirs, n_obj: int, scale: float) -> np.ndarray:
    dirs = _ensure_ref_dirs(ref_dirs, n_obj, n_points=200)
    den = np.sum(np.clip(dirs, 0.0, None), axis=1, keepdims=True)
    den = np.where(np.abs(den) <= 1e-12, 1.0, den)
    return float(scale) * np.clip(dirs, 0.0, None) / den


def _pf_positive_sphere(ref_dirs, n_obj: int) -> np.ndarray:
    dirs = np.clip(_ensure_ref_dirs(ref_dirs, n_obj, n_points=200), 0.0, None)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return dirs / norms


def _pf_dtlz56_curve(n_obj: int, n_points: int = 200) -> np.ndarray:
    n_obj = max(2, int(n_obj))
    n_points = max(32, int(n_points))
    x = np.linspace(0.0, 1.0, n_points, dtype=float)
    theta = np.full((n_points, n_obj - 1), 0.5, dtype=float)
    theta[:, 0] = x
    g = np.zeros(n_points, dtype=float)
    return _to_numpy(_obj_dtlz2_like(theta, n_obj, g, np))


def _pf_dtlz7_sample(n_obj: int, n_points: int = 400) -> np.ndarray:
    n_obj = max(2, int(n_obj))
    n_points = max(64, int(n_points))
    if n_obj == 2:
        f1 = np.linspace(0.0, 1.0, n_points, dtype=float)
        f2 = 2.0 * (2.0 - (f1 / 2.0) * (1.0 + np.sin(3.0 * np.pi * f1)))
        return np.column_stack([f1, f2])

    # Quasi-uniform random sample in [0,1]^(m-1) for a practical PF approximation.
    rng = np.random.default_rng(1)
    f_main = rng.random((n_points, n_obj - 1), dtype=float)
    h = n_obj - np.sum(f_main / 2.0 * (1.0 + np.sin(3.0 * np.pi * f_main)), axis=1)
    f_last = 2.0 * h
    return np.column_stack([f_main, f_last])


def _scale_objectives(F: np.ndarray, a: float) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    scale = float(a) ** np.arange(F.shape[1], dtype=float)
    return F * scale[None, :]


class _BaseDTLZ(Problem):
    _USE_JAX = False

    def _xp(self):
        return _xp(self._USE_JAX)

    def _calc_pareto_front(self, ref_dirs=None, n_pareto_points: int = 200, **kwargs):
        name = self.__class__.__name__.replace("_JAX", "").upper()
        m = int(getattr(self, "n_obj", 2))

        # Base/heuristic PFs for the DTLZ family.
        if "DTLZ1" in name:
            pf = _pf_simplex_scaled(ref_dirs, m, scale=0.5)
        elif any(tag in name for tag in ("DTLZ2", "DTLZ3", "DTLZ4")):
            pf = _pf_positive_sphere(ref_dirs, m)
        elif any(tag in name for tag in ("DTLZ5", "DTLZ6")):
            pf = _pf_dtlz56_curve(m, n_points=max(int(n_pareto_points), 200))
        elif "DTLZ7" in name:
            pf = _pf_dtlz7_sample(m, n_points=max(int(n_pareto_points), 400))
        else:
            return None

        # Objective-space transforms for DTLZ-derived variants (constraints ignored for PF-based metrics).
        if name.startswith("IDTLZ1"):
            return 0.5 - pf
        if name.startswith("IDTLZ2"):
            return 1.0 - pf
        if name.startswith("SDTLZ1") or name.startswith("SDTLZ2"):
            a = float(getattr(self, "a", 2.0))
            return _scale_objectives(pf, a)
        if name.startswith("CDTLZ2"):
            out = np.asarray(pf, dtype=float).copy()
            if out.shape[1] >= 2:
                out[:, :-1] = out[:, :-1] ** 4
                out[:, -1] = out[:, -1] ** 2
            return out
        return pf


class DTLZ1(_BaseDTLZ):
    """Ref: Deb et al., 2005."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 4
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 100.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        out["F"] = _to_numpy(_obj_dtlz1(x, self.n_obj, g, xp))


class DTLZ2(_BaseDTLZ):
    """Ref: Deb et al., 2005."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        g = xp.sum((x[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        out["F"] = _to_numpy(_obj_dtlz2_like(x, self.n_obj, g, xp))


class DTLZ3(_BaseDTLZ):
    """Ref: Deb et al., 2005."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 100.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        out["F"] = _to_numpy(_obj_dtlz2_like(x, self.n_obj, g, xp))


class DTLZ4(_BaseDTLZ):
    """Ref: Deb et al., 2005."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        x = x.copy()
        x = _assign(x, (slice(None), slice(0, self.n_obj - 1)), x[:, : self.n_obj - 1] ** 100, xp)
        g = xp.sum((x[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        out["F"] = _to_numpy(_obj_dtlz2_like(x, self.n_obj, g, xp))

class DTLZ5(_BaseDTLZ):
    """Ref: Deb et al., 2005."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        g = xp.sum((x[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        xe = x.copy()
        if self.n_obj > 2:
            tmp = xp.tile(g[:, None], (1, self.n_obj - 2))
            xe = _assign(
                xe,
                (slice(None), slice(1, self.n_obj - 1)),
                (1.0 + 2.0 * tmp * xe[:, 1 : self.n_obj - 1]) / (2.0 + 2.0 * tmp),
                xp,
            )
        out["F"] = _to_numpy(_obj_dtlz2_like(xe, self.n_obj, g, xp))


class DTLZ6(_BaseDTLZ):
    """Ref: Deb et al., 2005."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        g = xp.sum(x[:, self.n_obj - 1 :] ** 0.1, axis=1)
        xe = x.copy()
        if self.n_obj > 2:
            tmp = xp.tile(g[:, None], (1, self.n_obj - 2))
            xe = _assign(
                xe,
                (slice(None), slice(1, self.n_obj - 1)),
                (1.0 + 2.0 * tmp * xe[:, 1 : self.n_obj - 1]) / (2.0 + 2.0 * tmp),
                xp,
            )
        out["F"] = _to_numpy(_obj_dtlz2_like(xe, self.n_obj, g, xp))


class DTLZ7(_BaseDTLZ):
    """Ref: Deb et al., 2005."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 19
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        g = 1.0 + 9.0 * xp.mean(x[:, self.n_obj - 1 :], axis=1)
        f_main = x[:, : self.n_obj - 1]
        h = self.n_obj - xp.sum(
            f_main / (1.0 + g[:, None]) * (1.0 + xp.sin(3.0 * xp.pi * f_main)),
            axis=1,
        )
        f = xp.concatenate([f_main, ((1.0 + g) * h)[:, None]], axis=1)
        out["F"] = _to_numpy(f)


class DTLZ8(_BaseDTLZ):
    """Ref: Deb et al., 2005. Constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var_raw = int(n_var) if n_var is not None else 10 * n_obj
        n_var_adj = int(np.ceil(n_var_raw / n_obj) * n_obj)
        super().__init__(n_var=n_var_adj, n_obj=n_obj, n_ieq_constr=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        m = int(self.n_obj)
        step = self.n_var // m
        f_cols = []
        for i in range(m):
            lo = i * step
            hi = (i + 1) * step
            f_cols.append(xp.mean(x[:, lo:hi], axis=1))
        f = xp.column_stack(f_cols)

        g_main = 1.0 - f[:, [m - 1]] - 4.0 * f[:, : m - 1]
        if m == 2:
            g_last = xp.zeros_like(f[:, 0])
        else:
            mins = xp.sort(f[:, : m - 1], axis=1)
            g_last = 1.0 - 2.0 * f[:, m - 1] - xp.sum(mins[:, :2], axis=1)
        g = xp.column_stack([g_main, g_last])

        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(g)


class DTLZ9(_BaseDTLZ):
    """Ref: Deb et al., 2005. Constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 2, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var_raw = int(n_var) if n_var is not None else 10 * n_obj
        n_var_adj = int(np.ceil(n_var_raw / n_obj) * n_obj)
        super().__init__(
            n_var=n_var_adj,
            n_obj=n_obj,
            n_ieq_constr=max(1, n_obj - 1),
            xl=0.0,
            xu=1.0,
            vtype=float,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xp10 = x ** 0.1
        m = int(self.n_obj)
        step = self.n_var // m
        f_cols = []
        for i in range(m):
            lo = i * step
            hi = (i + 1) * step
            f_cols.append(xp.sum(xp10[:, lo:hi], axis=1))
        f = xp.column_stack(f_cols)

        c = 1.0 - f[:, [m - 1]] ** 2 - f[:, : m - 1] ** 2
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(c)


class IDTLZ1(_BaseDTLZ):
    """Ref: Jain and Deb, 2014."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 4
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 100.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        out["F"] = _to_numpy((1.0 + g)[:, None] / 2.0 - _obj_dtlz1(x, self.n_obj, g, xp))


class IDTLZ2(_BaseDTLZ):
    """Ref: Jain and Deb, 2014."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        g = xp.sum((x[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        out["F"] = _to_numpy((1.0 + g)[:, None] - _obj_dtlz2_like(x, self.n_obj, g, xp))


class SDTLZ1(_BaseDTLZ):
    """Ref: Deb and Jain, 2014."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, a: float = 2.0, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 4
        self.a = float(a)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 100.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz1(x, self.n_obj, g, xp)
        out["F"] = _to_numpy(f * (self.a ** xp.arange(self.n_obj))[None, :])


class SDTLZ2(_BaseDTLZ):
    """Ref: Deb and Jain, 2014."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, a: float = 2.0, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        self.a = float(a)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        g = xp.sum((x[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        f = _obj_dtlz2_like(x, self.n_obj, g, xp)
        out["F"] = _to_numpy(f * (self.a ** xp.arange(self.n_obj))[None, :])

class C1_DTLZ1(_BaseDTLZ):
    """Ref: Jain and Deb, 2014. Constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 4
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 100.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz1(x, self.n_obj, g, xp)
        c = f[:, -1] / 0.6 + xp.sum(f[:, :-1] / 0.5, axis=1) - 1.0
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(c[:, None])


class C1_DTLZ3(_BaseDTLZ):
    """Ref: Jain and Deb, 2014. Constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 10.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz2_like(x, self.n_obj, g, xp)
        if self.n_obj == 2:
            r = 6.0
        elif self.n_obj <= 3:
            r = 9.0
        elif self.n_obj <= 8:
            r = 12.5
        else:
            r = 15.0
        s = xp.sum(f ** 2, axis=1)
        c = -(s - 16.0) * (s - r**2)
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(c[:, None])


class C2_DTLZ2(_BaseDTLZ):
    """Ref: Jain and Deb, 2014. Constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        g = xp.sum((x[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        f = _obj_dtlz2_like(x, self.n_obj, g, xp)
        r = 0.4 if self.n_obj == 3 else 0.5
        s = xp.sum(f ** 2, axis=1)
        c1 = (f - 1.0) ** 2 + s[:, None] - f ** 2 - r**2
        c2 = xp.sum((f - 1.0 / xp.sqrt(float(self.n_obj))) ** 2, axis=1) - r**2
        c = xp.minimum(xp.min(c1, axis=1), c2)
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(c[:, None])


class C3_DTLZ4(_BaseDTLZ):
    """Ref: Jain and Deb, 2014. Constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xe = x.copy()
        xe = _assign(xe, (slice(None), slice(0, self.n_obj - 1)), xe[:, : self.n_obj - 1] ** 100, xp)
        g = xp.sum((xe[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        f = _obj_dtlz2_like(xe, self.n_obj, g, xp)
        s = xp.sum(f ** 2, axis=1)
        c = 1.0 - f ** 2 / 4.0 - (s[:, None] - f ** 2)
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(c)


class CDTLZ2(_BaseDTLZ):
    """Ref: Deb and Jain, 2014."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        g = xp.sum((x[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        f = _obj_dtlz2_like(x, self.n_obj, g, xp)
        out["F"] = _to_numpy(xp.column_stack([f[:, : self.n_obj - 1] ** 4, f[:, self.n_obj - 1] ** 2]))

class DC1_DTLZ1(_BaseDTLZ):
    """Ref: Li et al., 2018. Decision-space constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 4
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 100.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz1(x, self.n_obj, g, xp)
        c = 0.5 - xp.cos(3.0 * xp.pi * x[:, 0])
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(c[:, None])


class DC1_DTLZ3(_BaseDTLZ):
    """Ref: Li et al., 2018. Decision-space constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 10.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz2_like(x, self.n_obj, g, xp)
        c = 0.5 - xp.cos(3.0 * xp.pi * x[:, 0])
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(c[:, None])


class DC2_DTLZ1(_BaseDTLZ):
    """Ref: Li et al., 2018. Decision-space constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 4
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 100.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz1(x, self.n_obj, g, xp)
        c1 = 0.5 - xp.cos(3.0 * xp.pi * g)
        c2 = 0.5 - xp.exp(-g)
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(xp.column_stack([c1, c2]))


class DC2_DTLZ3(_BaseDTLZ):
    """Ref: Li et al., 2018. Decision-space constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 10.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz2_like(x, self.n_obj, g, xp)
        c1 = 0.5 - xp.cos(3.0 * xp.pi * g)
        c2 = 0.5 - xp.exp(-g)
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(xp.column_stack([c1, c2]))


class DC3_DTLZ1(_BaseDTLZ):
    """Ref: Li et al., 2018. Decision-space constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 4
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 100.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz1(x, self.n_obj, g, xp)
        c_dec = 0.5 - xp.cos(3.0 * xp.pi * x[:, : self.n_obj - 1])
        c_g = (0.5 - xp.cos(3.0 * xp.pi * g))[:, None]
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(xp.column_stack([c_dec, c_g]))


class DC3_DTLZ3(_BaseDTLZ):
    """Ref: Li et al., 2018. Decision-space constrained."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        xm = x[:, self.n_obj - 1 :]
        g = 10.0 * (self.n_var - self.n_obj + 1 + xp.sum((xm - 0.5) ** 2 - xp.cos(20.0 * xp.pi * (xm - 0.5)), axis=1))
        f = _obj_dtlz2_like(x, self.n_obj, g, xp)
        c_dec = 0.5 - xp.cos(3.0 * xp.pi * x[:, : self.n_obj - 1])
        c_g = (0.5 - xp.cos(3.0 * xp.pi * g))[:, None]
        out["F"] = _to_numpy(f)
        out["G"] = _to_numpy(xp.column_stack([c_dec, c_g]))


_CPU_CLASSES = [
    "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "DTLZ8", "DTLZ9",
    "IDTLZ1", "IDTLZ2", "SDTLZ1", "SDTLZ2",
    "C1_DTLZ1", "C1_DTLZ3", "C2_DTLZ2", "C3_DTLZ4", "CDTLZ2",
    "DC1_DTLZ1", "DC1_DTLZ3", "DC2_DTLZ1", "DC2_DTLZ3", "DC3_DTLZ1", "DC3_DTLZ3",
]

for _name in _CPU_CLASSES:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU_CLASSES + [f"{n}_JAX" for n in _CPU_CLASSES]
