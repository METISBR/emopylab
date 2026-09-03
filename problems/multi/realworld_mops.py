from __future__ import annotations

"""
Real-world and combinatorial multi/many-objective problems (community benchmarks).

Included
--------
- MLDMP
- MPDMP
- MOKP
- MONRP
- MOTSP
- mQAP

References (from original MATLAB sources)
-----------------------------------------
- MLDMP: M. Li et al., IEEE TEC, 2018.
- MPDMP: M. Koppen and K. Yoshida, EMO 2007.
- MOKP: E. Zitzler and L. Thiele, IEEE TEC, 1999.
- MONRP: Y. Zhang et al., GECCO 2007.
- MOTSP: D. Corne and J. Knowles, GECCO 2007.
- mQAP: J. Knowles and D. Corne, EMO 2003.
"""

from pathlib import Path

import numpy as np
from core.problem import Problem

try:
    from scipy.io import loadmat, savemat
    from scipy.stats import norm
except Exception:  # pragma: no cover - optional dependency at runtime
    loadmat = None
    savemat = None
    norm = None


def _require_scipy():
    if loadmat is None or savemat is None or norm is None:
        raise RuntimeError("scipy is required for realworld_mops problems (scipy.io + scipy.stats).")


def _as_2d(x, dtype=float):
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _module_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / "problems"
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        target = candidate / "Multi-objective optimization" / "Real-world MOPs"
        if target.is_dir():
            return target
    raise FileNotFoundError("MATLAB source folder for Real-world MOPs not found.")


def _rng():
    # Keep deterministic local generation when MATLAB .mat cache is absent.
    return np.random.default_rng(1)


def _line_intersection(p: np.ndarray) -> np.ndarray:
    if np.isclose(p[0, 0], p[1, 0]):
        x = p[0, 0]
        y = p[2, 1] + (x - p[2, 0]) * (p[2, 1] - p[3, 1]) / (p[2, 0] - p[3, 0] + 1e-30)
        return np.array([x, y], dtype=float)
    if np.isclose(p[2, 0], p[3, 0]):
        x = p[2, 0]
        y = p[0, 1] + (x - p[0, 0]) * (p[0, 1] - p[1, 1]) / (p[0, 0] - p[1, 0] + 1e-30)
        return np.array([x, y], dtype=float)
    k1 = (p[0, 1] - p[1, 1]) / (p[0, 0] - p[1, 0] + 1e-30)
    k2 = (p[2, 1] - p[3, 1]) / (p[2, 0] - p[3, 0] + 1e-30)
    x = (k1 * p[0, 0] - k2 * p[2, 0] + p[2, 1] - p[0, 1]) / (k1 - k2 + 1e-30)
    y = p[0, 1] + (x - p[0, 0]) * k1
    return np.array([x, y], dtype=float)


def _points_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    j = poly.shape[0] - 1
    for i in range(poly.shape[0]):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
        inside ^= intersect
        j = i
    return inside


def _point2line(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    num = np.abs(
        (line[0, 0] - pts[:, 0]) * (line[1, 1] - pts[:, 1]) - (line[1, 0] - pts[:, 0]) * (line[0, 1] - pts[:, 1])
    )
    den = np.sqrt((line[0, 0] - line[1, 0]) ** 2 + (line[0, 1] - line[1, 1]) ** 2)
    return num / max(den, 1e-30)


def _euclidean_cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a**2, axis=1, keepdims=True)
    bb = np.sum(b**2, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)
    return np.sqrt(d2)


def _repair_to_permutation(pop_dec: np.ndarray) -> np.ndarray:
    x = _as_2d(pop_dec, dtype=float).copy()
    d = x.shape[1]
    target = np.arange(1, d + 1, dtype=int)
    rounded = np.rint(x).astype(int)
    valid = np.all(np.sort(rounded, axis=1) == target[None, :], axis=1)
    if np.all(valid):
        return rounded
    ranks = np.argsort(x, axis=1, kind="mergesort") + 1
    rounded[~valid] = ranks[~valid]
    return rounded


def _load_or_create_mokp(m: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    _require_scipy()
    path = _module_dir() / f"MOKP-M{m}-D{d}.mat"
    if path.exists():
        data = loadmat(path)
        return np.asarray(data["P"], dtype=float), np.asarray(data["W"], dtype=float)
    rng = _rng()
    p = rng.integers(10, 101, size=(m, d)).astype(float)
    w = rng.integers(10, 101, size=(m, d)).astype(float)
    savemat(path, {"P": p, "W": w})
    return p, w


def _load_or_create_monrp(n: int, m_customers: int) -> tuple[np.ndarray, np.ndarray]:
    _require_scipy()
    path = _module_dir() / f"MONRP-n{n}-m{m_customers}.mat"
    if path.exists():
        data = loadmat(path)
        return np.asarray(data["Cost"], dtype=float).reshape(-1), np.asarray(data["Value"], dtype=float)
    rng = _rng()
    cost = rng.integers(1, 10, size=(n,)).astype(float)
    value = rng.integers(0, 6, size=(n, m_customers)).astype(float)
    savemat(path, {"Cost": cost[None, :], "Value": value})
    return cost, value


def _matlab_cell_to_list(cell_arr) -> list[np.ndarray]:
    arr = np.asarray(cell_arr)
    if arr.dtype != object:
        return [np.asarray(arr, dtype=float)]
    out: list[np.ndarray] = []
    for item in arr.reshape(-1):
        out.append(np.asarray(item, dtype=float))
    return out


def _load_or_create_motsp(m: int, d: int, c: float) -> list[np.ndarray]:
    _require_scipy()
    path = _module_dir() / f"MOTSP-M{m}-D{d}-c{c:.4f}.mat"
    if path.exists():
        data = loadmat(path)
        return _matlab_cell_to_list(data["C"])
    rng = _rng()
    c_mats = []
    c1 = rng.random((d, d))
    c_mats.append(c1)
    for _ in range(1, m):
        c_mats.append(c * c_mats[-1] + (1.0 - c) * rng.random((d, d)))
    for i in range(m):
        a = c_mats[i]
        c_mats[i] = np.tril(a, -1) + np.triu(a.T, 1)
    savemat(path, {"C": np.array(c_mats, dtype=object)[None, :]})
    return [np.asarray(v, dtype=float) for v in c_mats]


def _load_or_create_mqap(m: int, d: int, c: float) -> tuple[np.ndarray, list[np.ndarray]]:
    _require_scipy()
    path = _module_dir() / f"mQAP-M{m}-D{d}-c{c:.4f}.mat"
    if path.exists():
        data = loadmat(path)
        a = np.asarray(data["a"], dtype=float)
        b = _matlab_cell_to_list(data["b"])
        return a, b

    rng = _rng()
    a = rng.integers(1, 101, size=(d, d)).astype(float)
    a = a * (~np.eye(d, dtype=bool))
    rs = [rng.random((d, d)) for _ in range(m)]
    if m > 1:
        if c >= 0.9999:
            for i in range(1, m):
                rs[i] = rs[0].copy()
        elif c >= 0:
            base = rs[0]
            sigma = max(1e-12, 1.0 - np.sqrt(c))
            base_cdf1 = norm.cdf(1.0, loc=base, scale=sigma)
            base_cdf0 = norm.cdf(0.0, loc=base, scale=sigma)
            for i in range(1, m):
                s = rs[i]
                u = s * base_cdf1 + (1.0 - s) * base_cdf0
                rs[i] = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12), loc=base, scale=sigma)
        elif c > -0.9999:
            base = rs[0]
            sigma = max(1e-12, 1.0 - np.sqrt(-c))
            base_cdf1 = norm.cdf(1.0, loc=base, scale=sigma)
            base_cdf0 = norm.cdf(0.0, loc=base, scale=sigma)
            for i in range(1, m):
                s = rs[i]
                u = s * base_cdf1 + (1.0 - s) * base_cdf0
                rs[i] = 1.0 - norm.ppf(np.clip(u, 1e-12, 1 - 1e-12), loc=base, scale=sigma)
        else:
            for i in range(1, m):
                rs[i] = 1.0 - rs[0]

    b = [100.0 * r * (~np.eye(d, dtype=bool)) for r in rs]
    savemat(path, {"a": a, "b": np.array(b, dtype=object)[None, :]})
    return a, [np.asarray(v, dtype=float) for v in b]


class _BasePolygonDistanceProblem(Problem):
    def __init__(self, n_obj: int = 10, lower: float = -100.0, upper: float = 100.0, **kwargs):
        self._n_obj_req = max(3, int(n_obj))
        self._lower = float(lower)
        self._upper = float(upper)
        super().__init__(n_var=2, n_obj=self._n_obj_req, xl=np.array([lower, lower], dtype=float), xu=np.array([upper, upper], dtype=float), vtype=float, **kwargs)
        self.points = self._build_points(self.n_obj)

    @staticmethod
    def _build_points(m: int) -> np.ndarray:
        if m % 2 == 0:
            angle = (2.0 * np.arange(1, m + 1) - 3.0) * np.pi / m
        else:
            angle = (2.0 * np.arange(1, m + 1) - 2.0) * np.pi / m
        return np.column_stack([np.sin(angle), np.cos(angle)])


class MLDMP(_BasePolygonDistanceProblem):
    def __init__(self, n_obj: int = 10, lower: float = -100.0, upper: float = 100.0, **kwargs):
        super().__init__(n_obj=n_obj, lower=lower, upper=upper, **kwargs)
        m = self.n_obj
        heads = np.repeat(np.arange(1, m + 1), max(0, int(np.ceil(m / 2.0 - 2.0))))
        tails = np.tile(np.arange(1, max(0, int(np.ceil(m / 2.0 - 2.0))) + 1), m)
        tails = heads + tails
        self.polygons: list[np.ndarray] = []
        for h, t in zip(heads, tails):
            idx = (np.arange(h, t + 1) - 1) % m
            poly = self.points[idx, :]
            p_idx = np.array([h - 1, h, t, t + 1]) - 1
            p = self.points[p_idx % m, :]
            inter = _line_intersection(p)
            poly = np.vstack([poly, 2.0 * inter[None, :] - poly])
            self.polygons.append(poly)

    def _evaluate(self, x, out, *args, **kwargs):
        x = _as_2d(x, float)
        # MATLAB repairs invalid regions in CalDec. Here we evaluate directly for stability/determinism.
        f = np.zeros((x.shape[0], self.n_obj), dtype=float)
        for m in range(self.n_obj):
            line = self.points[np.mod([m, m + 1], self.n_obj), :]
            f[:, m] = _point2line(x, line)
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=400):
        g = int(np.ceil(np.sqrt(max(1, n_pareto_points))))
        X, Y = np.meshgrid(np.linspace(-1, 1, g), np.linspace(-1, 1, g))
        pts = np.column_stack([X.ravel(), Y.ravel()])
        valid = _points_in_poly(pts, self.points)
        for poly in getattr(self, "polygons", []):
            valid = valid & (~_points_in_poly(pts, poly))
        return self.evaluate(pts[valid], return_values_of=["F"])


class MPDMP(_BasePolygonDistanceProblem):
    def __init__(self, n_obj: int = 10, lower: float = -100.0, upper: float = 100.0, **kwargs):
        super().__init__(n_obj=n_obj, lower=lower, upper=upper, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = _as_2d(x, float)
        out["F"] = _euclidean_cdist(x, self.points)

    def _calc_pareto_front(self, n_pareto_points=400):
        g = int(np.ceil(np.sqrt(max(1, n_pareto_points))))
        X, Y = np.meshgrid(np.linspace(-1, 1, g), np.linspace(-1, 1, g))
        pts = np.column_stack([X.ravel(), Y.ravel()])
        valid = _points_in_poly(pts, self.points)
        return _euclidean_cdist(pts[valid], self.points)


class MOKP(Problem):
    def __init__(self, n_var: int = 250, n_obj: int = 2, **kwargs):
        n_var = int(n_var)
        n_obj = int(max(2, n_obj))
        self.P, self.W = _load_or_create_mokp(n_obj, n_var)
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_obj, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = (np.asarray(_as_2d(x, float)) >= 0.5).astype(float)
        out["F"] = np.sum(self.P, axis=1)[None, :] - x @ self.P.T
        out["G"] = x @ self.W.T - (np.sum(self.W, axis=1) / 2.0)[None, :]


class MONRP(Problem):
    def __init__(self, n_var: int = 100, m_customers: int = 100, **kwargs):
        self.m_customers = int(m_customers)
        n_var = int(n_var)
        self.Cost, self.Value = _load_or_create_monrp(n_var, self.m_customers)
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = (np.asarray(_as_2d(x, float)) >= 0.5).astype(float)
        f1 = np.sum(self.Cost[None, :] * x, axis=1)
        f2 = np.sum(self.Value) - np.sum(x @ self.Value, axis=1)
        out["F"] = np.column_stack([f1, f2])


class MOTSP(Problem):
    def __init__(self, n_var: int = 30, n_obj: int = 2, c: float = 0.0, **kwargs):
        self.c = float(c)
        n_var = int(n_var)
        n_obj = int(max(2, n_obj))
        self.C = _load_or_create_motsp(n_obj, n_var, self.c)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=1.0, xu=float(n_var), vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        perm = _repair_to_permutation(x)
        n, d = perm.shape
        f = np.zeros((n, self.n_obj), dtype=float)
        idx_from = perm - 1
        idx_to = np.roll(idx_from, -1, axis=1)
        rows = np.arange(n)[:, None]
        for j in range(self.n_obj):
            cmat = self.C[j]
            f[:, j] = np.sum(cmat[idx_from[rows, np.arange(d)], idx_to[rows, np.arange(d)]], axis=1)
        out["F"] = f


class mQAP(Problem):
    def __init__(self, n_var: int = 10, n_obj: int = 2, c: float = 0.0, **kwargs):
        self.c = float(c)
        n_var = int(n_var)
        n_obj = int(max(2, n_obj))
        self.a, self.b = _load_or_create_mqap(n_obj, n_var, self.c)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=1.0, xu=float(n_var), vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        perm = _repair_to_permutation(x)
        n = perm.shape[0]
        f = np.zeros((n, self.n_obj), dtype=float)
        idx = perm - 1
        for i in range(n):
            p = idx[i]
            for j in range(self.n_obj):
                f[i, j] = np.sum(self.a * self.b[j][np.ix_(p, p)])
        out["F"] = f


_CPU = ["MLDMP", "MPDMP", "MOKP", "MONRP", "MOTSP", "mQAP"]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
