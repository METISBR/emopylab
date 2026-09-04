from __future__ import annotations

"""
MaF benchmark family converted for local use in EmoPyLab.

Reference
---------
R. Cheng, M. Li, Y. Tian, X. Zhang, S. Yang, Y. Jin, and X. Yao.
A benchmark test suite for evolutionary many-objective optimization.
Complex & Intelligent Systems, 2017, 3(1): 67-81.
"""

import math

import numpy as np

from core.problem import Problem


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _uniform_simplex(n: int, m: int, seed: int = 1) -> np.ndarray:
    n = max(1, int(n))
    m = max(2, int(m))
    if m == 2:
        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([x, 1.0 - x])
    rng = np.random.default_rng(seed)
    w = rng.random((n, m))
    w /= np.maximum(np.sum(w, axis=1, keepdims=True), 1e-30)
    return w


def _uniform_box(n: int, d: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((max(1, int(n)), max(1, int(d))))


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
            continue
        keep &= ~(np.all(f[i] <= f, axis=1) & np.any(f[i] < f, axis=1))
        keep[i] = True
    return keep


def _shape_linear(x: np.ndarray, m: int) -> np.ndarray:
    n = x.shape[0]
    left = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), x[:, : m - 1]]), axis=1))
    right = np.column_stack([np.ones((n, 1)), 1.0 - x[:, m - 2 :: -1]])
    return left * right


def _shape_dtlz2(x: np.ndarray, m: int) -> np.ndarray:
    n = x.shape[0]
    left = np.fliplr(
        np.cumprod(
            np.column_stack([np.ones((n, 1)), np.cos(x[:, : m - 1] * np.pi / 2.0)]),
            axis=1,
        )
    )
    right = np.column_stack([np.ones((n, 1)), np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return left * right


def _shape_convex(x: np.ndarray) -> np.ndarray:
    return np.fliplr(
        np.cumprod(
            np.column_stack([np.ones((x.shape[0], 1)), 1.0 - np.cos(x[:, :-1] * np.pi / 2.0)]),
            axis=1,
        )
    ) * np.column_stack([np.ones((x.shape[0], 1)), 1.0 - np.sin(x[:, -2::-1] * np.pi / 2.0)])


def _shape_concave(x: np.ndarray) -> np.ndarray:
    return np.fliplr(
        np.cumprod(
            np.column_stack([np.ones((x.shape[0], 1)), np.sin(x[:, :-1] * np.pi / 2.0)]),
            axis=1,
        )
    ) * np.column_stack([np.ones((x.shape[0], 1)), np.cos(x[:, -2::-1] * np.pi / 2.0)])


def _wfg_convex_reference_x(n: int, m: int, kind: str) -> np.ndarray:
    r = _uniform_simplex(n, m)
    c = np.ones((r.shape[0], m), dtype=float)
    for i in range(r.shape[0]):
        for j in range(2, m + 1):
            product = np.prod(1.0 - c[i, m - j + 1 : m - 1]) if (m - j + 1) < (m - 1) else 1.0
            temp = r[i, j - 1] / max(r[i, 0], 1e-30) * product
            temp = max(float(temp), 0.0)
            c[i, m - j] = (temp**2 - temp + math.sqrt(2.0 * temp)) / (temp**2 + 1.0)

    x = np.arccos(np.clip(c, -1.0, 1.0)) * 2.0 / np.pi
    temp = (1.0 - np.sin(np.pi / 2.0 * x[:, 1])) * r[:, -1] / np.maximum(r[:, -2], 1e-30)
    a = np.linspace(0.0, 1.0, 10001)
    if kind == "disc":
        base = a * np.cos(5.0 * np.pi * a) ** 2
    else:
        base = a + np.cos(10.0 * np.pi * a + np.pi / 2.0) / (10.0 * np.pi)
    error = np.abs(temp[:, None] * (1.0 - np.cos(np.pi / 2.0 * a))[None, :] - 1.0 + base[None, :])
    top = min(10, a.size)
    nearest = np.argpartition(error, top - 1, axis=1)[:, :top]
    x[:, 0] = a[np.min(nearest, axis=1)]
    return x


def _s_linear(y: np.ndarray, a: float) -> np.ndarray:
    return np.abs(y - a) / np.abs(np.floor(a - y) + a)


def _b_flat(y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    out = a + np.minimum(0.0, np.floor(y - b)) * a * (b - y) / b
    out -= np.minimum(0.0, np.floor(c - y)) * (1.0 - a) * (y - c) / (1.0 - c)
    return np.round(out * 1e4) / 1e4


def _r_sum(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sum(y * w[None, :], axis=1) / np.sum(w)


def _r_nonsep(y: np.ndarray, a: int) -> np.ndarray:
    a = int(max(1, a))
    n, d = y.shape
    out = np.zeros(n, dtype=float)
    for j in range(d):
        temp = np.zeros(n, dtype=float)
        for k in range(a - 1):
            temp += np.abs(y[:, j] - y[:, (j + k + 1) % d])
        out += y[:, j] + temp
    den = (d / a) * math.ceil(a / 2.0) * (1.0 + 2.0 * a - 2.0 * math.ceil(a / 2.0))
    return out / den


def _s_decept(y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return 1.0 + (np.abs(y - a) - b) * (
        np.floor(y - a + b) * (1.0 - c + (a - b) / b) / (a - b)
        + np.floor(a + b - y) * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b)
        + 1.0 / b
    )


def _s_multi(y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    u = np.abs(y - c) / 2.0 / (np.floor(c - y) + c)
    return (1.0 + np.cos((4.0 * a + 2.0) * np.pi * (0.5 - u)) + 4.0 * b * u**2) / (b + 2.0)


def _rastrigin(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)


def _rosenbrock(x: np.ndarray) -> np.ndarray:
    if x.shape[1] < 2:
        return np.zeros(x.shape[0], dtype=float)
    return np.sum(100.0 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def _griewank(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    j = np.sqrt(np.arange(1, x.shape[1] + 1, dtype=float))[None, :]
    return np.sum(x**2, axis=1) / 4000.0 - np.prod(np.cos(x / j), axis=1) + 1.0


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def _line_intersection(p: np.ndarray) -> np.ndarray:
    if np.isclose(p[0, 0], p[1, 0]):
        x = p[0, 0]
        y = p[2, 1] + (x - p[2, 0]) * (p[2, 1] - p[3, 1]) / (p[2, 0] - p[3, 0])
        return np.array([x, y], dtype=float)
    if np.isclose(p[2, 0], p[3, 0]):
        x = p[2, 0]
        y = p[0, 1] + (x - p[0, 0]) * (p[0, 1] - p[1, 1]) / (p[0, 0] - p[1, 0])
        return np.array([x, y], dtype=float)
    k1 = (p[0, 1] - p[1, 1]) / (p[0, 0] - p[1, 0])
    k2 = (p[2, 1] - p[3, 1]) / (p[2, 0] - p[3, 0])
    x = (k1 * p[0, 0] - k2 * p[2, 0] + p[2, 1] - p[0, 1]) / (k1 - k2)
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


def _point_line_distance(pop_dec: np.ndarray, line: np.ndarray) -> np.ndarray:
    num = np.abs(
        (line[0, 0] - pop_dec[:, 0]) * (line[1, 1] - pop_dec[:, 1])
        - (line[1, 0] - pop_dec[:, 0]) * (line[0, 1] - pop_dec[:, 1])
    )
    den = np.sqrt((line[0, 0] - line[1, 0]) ** 2 + (line[0, 1] - line[1, 1]) ** 2)
    return num / max(den, 1e-30)


class _BaseMaF(Problem):
    def _clip(self, x) -> np.ndarray:
        return np.clip(_as_2d(x), self.xl, self.xu)


class MaF1(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m = self.n_obj
        g = np.sum((x[:, m - 1 :] - 0.5) ** 2, axis=1)
        h = _shape_linear(x, m)
        out["F"] = (1.0 + g)[:, None] - (1.0 + g)[:, None] * h

    def _calc_pareto_front(self, n_pareto_points=200):
        return 1.0 - _uniform_simplex(n_pareto_points, self.n_obj)


class MaF2(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        m = self.n_obj

        g = np.zeros((n, m), dtype=float)
        block = int(math.floor((d - m + 1) / m))
        for mi in range(m):
            if mi < m - 1:
                lo = (m - 1) + mi * block
                hi = (m - 1) + (mi + 1) * block
            else:
                lo = (m - 1) + (m - 1) * block
                hi = d
            seg = (x[:, lo:hi] / 2.0 + 0.25) - 0.5
            g[:, mi] = np.sum(seg**2, axis=1)

        xx = x.copy()
        xx[:, : m - 1] = xx[:, : m - 1] / 2.0 + 0.25
        out["F"] = (1.0 + g) * _shape_dtlz2(xx, m)

    def _calc_pareto_front(self, n_pareto_points=200):
        m = self.n_obj
        r = _uniform_simplex(n_pareto_points, m)
        c = np.ones((r.shape[0], m - 1), dtype=float)
        for i in range(r.shape[0]):
            for j in range(2, m + 1):
                if m - j + 2 <= m - 1:
                    prod_term = np.prod(c[i, (m - j + 1) : (m - 1)])
                else:
                    prod_term = 1.0
                temp = r[i, j - 1] / max(r[i, 0], 1e-30) * prod_term
                c[i, m - j] = np.sqrt(1.0 / (1.0 + temp**2))

        if m > 5:
            c = c * (np.cos(np.pi / 8.0) - np.cos(3.0 * np.pi / 8.0)) + np.cos(3.0 * np.pi / 8.0)
        else:
            lb = np.cos(3.0 * np.pi / 8.0)
            ub = np.cos(np.pi / 8.0)
            c = c[np.all((c >= lb) & (c <= ub), axis=1)]
            if c.size == 0:
                return np.empty((0, m), dtype=float)

        return np.fliplr(np.cumprod(np.column_stack([np.ones((c.shape[0], 1)), c]), axis=1)) * np.column_stack(
            [np.ones((c.shape[0], 1)), np.sqrt(1.0 - c[:, -1::-1] ** 2)]
        )


class MaF3(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m = self.n_obj
        g = 100.0 * (
            self.n_var - m + 1 + np.sum((x[:, m - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[:, m - 1 :] - 0.5)), axis=1)
        )
        f = (1.0 + g)[:, None] * _shape_dtlz2(x, m)
        f[:, : m - 1] = f[:, : m - 1] ** 4
        f[:, m - 1] = f[:, m - 1] ** 2
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj) ** 2
        temp = np.sum(np.sqrt(r[:, :-1]), axis=1) + r[:, -1]
        return r / np.column_stack([np.tile(temp[:, None] ** 2, (1, self.n_obj - 1)), temp])


class MaF4(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m = self.n_obj
        g = 100.0 * (
            self.n_var - m + 1 + np.sum((x[:, m - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[:, m - 1 :] - 0.5)), axis=1)
        )
        f = (1.0 + g)[:, None] - (1.0 + g)[:, None] * _shape_dtlz2(x, m)
        f *= (2.0 ** np.arange(1, m + 1))[None, :]
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return (1.0 - r) * (2.0 ** np.arange(1, self.n_obj + 1))[None, :]


class MaF5(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m = self.n_obj
        xx = x.copy()
        xx[:, : m - 1] = xx[:, : m - 1] ** 100
        g = np.sum((xx[:, m - 1 :] - 0.5) ** 2, axis=1)
        f = (1.0 + g)[:, None] * _shape_dtlz2(xx, m)
        f *= (2.0 ** np.arange(m, 0, -1))[None, :]
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return r * (2.0 ** np.arange(self.n_obj, 0, -1))[None, :]


class MaF6(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m = self.n_obj
        i = 2
        g = np.sum((x[:, m - 1 :] - 0.5) ** 2, axis=1)
        xx = x.copy()
        if m - i > 0:
            temp = np.tile(g[:, None], (1, m - i))
            xx[:, i - 1 : m - 1] = (1.0 + 2.0 * temp * xx[:, i - 1 : m - 1]) / (2.0 + 2.0 * temp)
        out["F"] = (1.0 + 100.0 * g)[:, None] * _shape_dtlz2(xx, m)

    def _calc_pareto_front(self, n_pareto_points=200):
        m = self.n_obj
        i = 2
        r = _uniform_simplex(n_pareto_points, i)
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        if m - r.shape[1] > 0:
            r = np.hstack([np.repeat(r[:, [0]], m - r.shape[1], axis=1), r])
        exponents = np.maximum(np.array([m - i] + list(range(m - i, 1 - i, -1)), dtype=float), 0.0)
        return r / (np.sqrt(2.0) ** exponents[None, :])


class MaF7(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 19
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        m = self.n_obj
        g = 1.0 + 9.0 * np.mean(x[:, m - 1 :], axis=1)
        f = np.zeros((x.shape[0], m), dtype=float)
        f[:, : m - 1] = x[:, : m - 1]
        f[:, m - 1] = (1.0 + g) * (
            m
            - np.sum(
                f[:, : m - 1] / (1.0 + g[:, None]) * (1.0 + np.sin(3.0 * np.pi * f[:, : m - 1])),
                axis=1,
            )
        )
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=200):
        m = self.n_obj
        interval = np.array([0.0, 0.251412, 0.631627, 0.859401], dtype=float)
        median = (interval[1] - interval[0]) / ((interval[3] - interval[2]) + (interval[1] - interval[0]))
        x = _uniform_box(n_pareto_points, m - 1)
        mask = x <= median
        x[mask] = x[mask] * (interval[1] - interval[0]) / median + interval[0]
        x[~mask] = (x[~mask] - median) * (interval[3] - interval[2]) / (1.0 - median) + interval[2]
        f_m = 2.0 * (m - np.sum(x / 2.0 * (1.0 + np.sin(3.0 * np.pi * x)), axis=1))
        return np.column_stack([x, f_m])


class MaF8(_BaseMaF):
    def __init__(self, n_obj: int = 10, **kwargs):
        n_obj = max(3, int(n_obj))
        self.points = self._build_points(n_obj)
        super().__init__(n_var=2, n_obj=n_obj, xl=np.array([-10000.0, -10000.0]), xu=np.array([10000.0, 10000.0]), vtype=float, **kwargs)

    @staticmethod
    def _build_points(m: int) -> np.ndarray:
        theta0 = np.pi / 2.0
        angles = theta0 - np.arange(1, m + 1, dtype=float) * 2.0 * np.pi / m
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        diff = x[:, None, :] - self.points[None, :, :]
        out["F"] = np.sqrt(np.sum(diff**2, axis=2))

    def _calc_pareto_front(self, n_pareto_points=200):
        g = int(np.ceil(np.sqrt(max(4, int(n_pareto_points)))))
        xs = np.linspace(-1.0, 1.0, g)
        xx, yy = np.meshgrid(xs, xs)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        in_poly = _points_in_poly(pts, self.points)
        pts = pts[in_poly]
        if pts.size == 0:
            return np.empty((0, self.n_obj), dtype=float)
        diff = pts[:, None, :] - self.points[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))


class MaF9(_BaseMaF):
    def __init__(self, n_obj: int = 10, **kwargs):
        n_obj = max(3, int(n_obj))
        self.points = MaF8._build_points(n_obj)
        self.polygons = self._build_polygons(self.points)
        super().__init__(n_var=2, n_obj=n_obj, xl=np.array([-10000.0, -10000.0]), xu=np.array([10000.0, 10000.0]), vtype=float, **kwargs)

    @staticmethod
    def _build_polygons(points: np.ndarray) -> list[np.ndarray]:
        m = points.shape[0]
        span = int(np.ceil(m / 2.0 - 2.0))
        polys: list[np.ndarray] = []
        if span <= 0:
            return polys

        head = np.repeat(np.arange(1, m + 1, dtype=int), span)
        tail = np.tile(np.arange(1, span + 1, dtype=int), m).reshape(-1) + head
        for h, t in zip(head, tail, strict=True):
            ids = (np.arange(h, t + 1, dtype=int) - 1) % m
            poly = points[ids]
            idx = (np.array([h - 1, h, t, t + 1], dtype=int) - 1) % m
            inter = _line_intersection(points[idx])
            poly2 = 2.0 * inter[None, :] - poly
            polys.append(np.vstack([poly, poly2]))
        return polys

    def _invalid_mask(self, x: np.ndarray) -> np.ndarray:
        invalid = np.zeros(x.shape[0], dtype=bool)
        for poly in self.polygons:
            invalid |= _points_in_poly(x, poly)
        outer = _points_in_poly(x, self.points)
        return invalid & ~outer

    def _repair(self, x: np.ndarray) -> np.ndarray:
        invalid = self._invalid_mask(x)
        if not np.any(invalid):
            return x
        rng = np.random.default_rng(123)
        repaired = x.copy()
        for _ in range(64):
            if not np.any(invalid):
                break
            repaired[invalid] = rng.uniform(self.xl, self.xu, size=(int(np.sum(invalid)), self.n_var))
            invalid = self._invalid_mask(repaired)
        repaired[invalid] = 0.0
        return repaired

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._repair(self._clip(x))
        f = np.zeros((x.shape[0], self.n_obj), dtype=float)
        for m in range(self.n_obj):
            line = self.points[[m, (m + 1) % self.n_obj], :]
            f[:, m] = _point_line_distance(x, line)
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=200):
        g = int(np.ceil(np.sqrt(max(4, int(n_pareto_points)))))
        xs = np.linspace(-1.0, 1.0, g)
        xx, yy = np.meshgrid(xs, xs)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        in_poly = _points_in_poly(pts, self.points)
        pts = pts[in_poly]
        if pts.size == 0:
            return np.empty((0, self.n_obj), dtype=float)
        f = np.zeros((pts.shape[0], self.n_obj), dtype=float)
        for m in range(self.n_obj):
            line = self.points[[m, (m + 1) % self.n_obj], :]
            f[:, m] = _point_line_distance(pts, line)
        return f


class MaF10(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        xl = np.zeros(n_var, dtype=float)
        xu = 2.0 * np.arange(1, n_var + 1, dtype=float)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        pop_dec = self._clip(x)
        n, d = pop_dec.shape
        m = self.n_obj
        k = m - 1
        l = d - k
        s = np.arange(2.0, 2.0 * m + 1.0, 2.0)

        z01 = pop_dec / (2.0 * np.arange(1, d + 1, dtype=float))[None, :]

        t1 = np.zeros((n, k + l), dtype=float)
        t1[:, :k] = z01[:, :k]
        t1[:, k:] = _s_linear(z01[:, k:], 0.35)

        t2 = np.zeros((n, k + l), dtype=float)
        t2[:, :k] = t1[:, :k]
        t2[:, k:] = _b_flat(t1[:, k:], 0.8, 0.75, 0.85)

        t3 = t2**0.02

        t4 = np.zeros((n, m), dtype=float)
        for i in range(1, m):
            lo = int((i - 1) * k / (m - 1))
            hi = int(i * k / (m - 1))
            w = np.arange(2 * lo + 2, 2 * hi + 1, 2, dtype=float)
            t4[:, i - 1] = _r_sum(t3[:, lo:hi], w)
        t4[:, m - 1] = _r_sum(t3[:, k : k + l], np.arange(2 * (k + 1), 2 * (k + l) + 1, 2, dtype=float))

        xx = np.zeros((n, m), dtype=float)
        for i in range(m - 1):
            xx[:, i] = np.maximum(t4[:, m - 1], 1.0) * (t4[:, i] - 0.5) + 0.5
        xx[:, m - 1] = t4[:, m - 1]

        h = _shape_convex(xx)
        h[:, m - 1] = 1.0 - xx[:, 0] - np.cos(10.0 * np.pi * xx[:, 0] + np.pi / 2.0) / (10.0 * np.pi)
        out["F"] = xx[:, [m - 1]] + s[None, :] * h

    def _calc_pareto_front(self, n_pareto_points=200):
        m = self.n_obj
        x = _wfg_convex_reference_x(n_pareto_points, m, "mixed")
        r = _shape_convex(x)
        r[:, m - 1] = 1.0 - x[:, 0] - np.cos(10.0 * np.pi * x[:, 0] + np.pi / 2.0) / (10.0 * np.pi)
        return (2.0 * np.arange(1, m + 1, dtype=float))[None, :] * r


class MaF11(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        d0 = int(n_var) if n_var is not None else n_obj + 9
        n_var = int(np.ceil((d0 - n_obj + 1) / 2.0) * 2 + n_obj - 1)
        xl = np.zeros(n_var, dtype=float)
        xu = 2.0 * np.arange(1, n_var + 1, dtype=float)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        pop_dec = self._clip(x)
        n, d = pop_dec.shape
        m = self.n_obj
        k = m - 1
        l = d - k
        s = np.arange(2.0, 2.0 * m + 1.0, 2.0)

        z01 = pop_dec / (2.0 * np.arange(1, d + 1, dtype=float))[None, :]

        t1 = np.zeros((n, k + l), dtype=float)
        t1[:, :k] = z01[:, :k]
        t1[:, k:] = _s_linear(z01[:, k:], 0.35)

        t2 = np.zeros((n, k + l // 2), dtype=float)
        t2[:, :k] = t1[:, :k]
        t2[:, k:] = (t1[:, k::2] + t1[:, k + 1 :: 2] + 2.0 * np.abs(t1[:, k::2] - t1[:, k + 1 :: 2])) / 3.0

        t3 = np.zeros((n, m), dtype=float)
        for i in range(1, m):
            lo = int((i - 1) * k / (m - 1))
            hi = int(i * k / (m - 1))
            t3[:, i - 1] = _r_sum(t2[:, lo:hi], np.ones(hi - lo, dtype=float))
        t3[:, m - 1] = _r_sum(t2[:, k : k + l // 2], np.ones(l // 2, dtype=float))

        xx = np.zeros((n, m), dtype=float)
        for i in range(m - 1):
            xx[:, i] = np.maximum(t3[:, m - 1], 1.0) * (t3[:, i] - 0.5) + 0.5
        xx[:, m - 1] = t3[:, m - 1]

        h = _shape_convex(xx)
        h[:, m - 1] = 1.0 - xx[:, 0] * np.cos(5.0 * np.pi * xx[:, 0]) ** 2
        f = xx[:, [m - 1]] + s[None, :] * h
        out["F"] = f[_nondominated_mask(f)] if f.shape[0] <= 2 else f

    def _calc_pareto_front(self, n_pareto_points=200):
        m = self.n_obj
        x = _wfg_convex_reference_x(n_pareto_points, m, "disc")
        r = _shape_convex(x)
        r[:, m - 1] = 1.0 - x[:, 0] * np.cos(5.0 * np.pi * x[:, 0]) ** 2
        r = r[_nondominated_mask(r)]
        return (2.0 * np.arange(1, m + 1, dtype=float))[None, :] * r


class MaF12(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else n_obj + 9
        xl = np.zeros(n_var, dtype=float)
        xu = 2.0 * np.arange(1, n_var + 1, dtype=float)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        pop_dec = self._clip(x)
        n, d = pop_dec.shape
        m = self.n_obj
        k = m - 1
        l = d - k
        s = np.arange(2.0, 2.0 * m + 1.0, 2.0)

        z01 = pop_dec / (2.0 * np.arange(1, d + 1, dtype=float))[None, :]
        suffix = np.fliplr(np.cumsum(np.fliplr(z01), axis=1)) - z01
        y = np.zeros_like(z01)
        if d > 1:
            y[:, : d - 1] = suffix[:, : d - 1] / np.arange(d - 1, 0, -1, dtype=float)[None, :]

        t1 = np.zeros((n, k + l), dtype=float)
        t1[:, : d - 1] = z01[:, : d - 1] ** (
            0.02
            + (50.0 - 0.02)
            * (0.98 / 49.98 - (1.0 - 2.0 * y[:, : d - 1]) * np.abs(np.floor(0.5 - y[:, : d - 1]) + 0.98 / 49.98))
        )
        t1[:, -1] = z01[:, -1]

        t2 = np.zeros((n, k + l), dtype=float)
        t2[:, :k] = _s_decept(t1[:, :k], 0.35, 0.001, 0.05)
        t2[:, k:] = _s_multi(t1[:, k:], 30.0, 95.0, 0.35)

        t3 = np.zeros((n, m), dtype=float)
        for i in range(1, m):
            lo = int((i - 1) * k / (m - 1))
            hi = int(i * k / (m - 1))
            t3[:, i - 1] = _r_nonsep(t2[:, lo:hi], int(k / (m - 1)))
        t3[:, m - 1] = _r_nonsep(t2[:, k:], l)

        xx = np.zeros((n, m), dtype=float)
        for i in range(m - 1):
            xx[:, i] = np.maximum(t3[:, m - 1], 1.0) * (t3[:, i] - 0.5) + 0.5
        xx[:, m - 1] = t3[:, m - 1]

        out["F"] = xx[:, [m - 1]] + s[None, :] * _shape_concave(xx)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return (2.0 * np.arange(1, self.n_obj + 1, dtype=float))[None, :] * r


class MaF13(_BaseMaF):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(3, int(n_obj))
        n_var = int(n_var) if n_var is not None else 5
        n_var = max(5, n_var)
        xl = np.concatenate([np.zeros(2), -2.0 * np.ones(n_var - 2)])
        xu = np.concatenate([np.ones(2), 2.0 * np.ones(n_var - 2)])
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n, d = x.shape
        idx = np.arange(1, d + 1, dtype=float)[None, :]
        y = x - 2.0 * x[:, [1]] * np.sin(2.0 * np.pi * x[:, [0]] + idx * np.pi / d)

        def _safe_mean(v: np.ndarray) -> np.ndarray:
            if v.shape[1] == 0:
                return np.zeros(v.shape[0], dtype=float)
            return np.mean(v, axis=1)

        f1 = np.sin(x[:, 0] * np.pi / 2.0) + 2.0 * _safe_mean(y[:, 3::3] ** 2)
        f2 = np.cos(x[:, 0] * np.pi / 2.0) * np.sin(x[:, 1] * np.pi / 2.0) + 2.0 * _safe_mean(y[:, 4::3] ** 2)
        f3 = np.cos(x[:, 0] * np.pi / 2.0) * np.cos(x[:, 1] * np.pi / 2.0) + 2.0 * _safe_mean(y[:, 2::3] ** 2)

        f = np.zeros((n, self.n_obj), dtype=float)
        f[:, 0] = f1
        f[:, 1] = f2
        f[:, 2] = f3
        tail_mean = _safe_mean(y[:, 3:] ** 2)
        if self.n_obj > 3:
            f[:, 3:] = (f1**2 + f2**10 + f3**10 + 2.0 * tail_mean)[:, None]
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, 3)
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        if self.n_obj > 3:
            ext = (r[:, 0] ** 2 + r[:, 1] ** 10 + r[:, 2] ** 10)[:, None]
            r = np.hstack([r, np.tile(ext, (1, self.n_obj - 3))])
        return r


class _LSMOPLike(Problem):
    _INVERT = False

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = max(2, int(n_obj))
        n_var = int(n_var) if n_var is not None else 20 * n_obj
        self.nk = 2

        c = [3.8 * 0.1 * (1.0 - 0.1)]
        for _ in range(n_obj - 1):
            c.append(3.8 * c[-1] * (1.0 - c[-1]))
        c = np.asarray(c, dtype=float)
        sublen = np.floor(c / np.sum(c) * (n_var - n_obj + 1) / self.nk).astype(int)
        sublen = np.maximum(sublen, 1)
        self.sublen = sublen
        self.len = np.concatenate([[0], np.cumsum(self.sublen * self.nk)])

        xl = np.zeros(n_var, dtype=float)
        xu = np.concatenate([np.ones(n_obj - 1), 10.0 * np.ones(n_var - n_obj + 1)])
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _clip(self, x):
        arr = _as_2d(x)
        return np.clip(arr, self.xl, self.xu)


class MaF14(_LSMOPLike):
    def _evaluate(self, x, out, *args, **kwargs):
        pop_dec = self._clip(x)
        n, d = pop_dec.shape
        m = self.n_obj

        xx = pop_dec.copy()
        scale = 1.0 + np.arange(m, d + 1, dtype=float) / d
        xx[:, m - 1 :] = scale[None, :] * xx[:, m - 1 :] - xx[:, [0]] * 10.0

        g = np.zeros((n, m), dtype=float)
        for i in range(0, m, 2):
            for j in range(self.nk):
                lo = self.len[i] + m - 1 + j * self.sublen[i]
                hi = self.len[i] + m - 1 + (j + 1) * self.sublen[i]
                g[:, i] += _rastrigin(xx[:, lo:hi])
        for i in range(1, m, 2):
            for j in range(self.nk):
                lo = self.len[i] + m - 1 + j * self.sublen[i]
                hi = self.len[i] + m - 1 + (j + 1) * self.sublen[i]
                g[:, i] += _rosenbrock(xx[:, lo:hi])

        g = g / np.maximum(self.sublen[None, :], 1) / self.nk
        out["F"] = (1.0 + g) * _shape_linear(pop_dec, m)

    def _calc_pareto_front(self, n_pareto_points=200):
        return _uniform_simplex(n_pareto_points, self.n_obj)


class MaF15(_LSMOPLike):
    def _evaluate(self, x, out, *args, **kwargs):
        pop_dec = self._clip(x)
        n, d = pop_dec.shape
        m = self.n_obj

        xx = pop_dec.copy()
        scale = 1.0 + np.cos(np.arange(m, d + 1, dtype=float) / d * np.pi / 2.0)
        xx[:, m - 1 :] = scale[None, :] * xx[:, m - 1 :] - xx[:, [0]] * 10.0

        g = np.zeros((n, m), dtype=float)
        for i in range(0, m, 2):
            for j in range(self.nk):
                lo = self.len[i] + m - 1 + j * self.sublen[i]
                hi = self.len[i] + m - 1 + (j + 1) * self.sublen[i]
                g[:, i] += _griewank(xx[:, lo:hi])
        for i in range(1, m, 2):
            for j in range(self.nk):
                lo = self.len[i] + m - 1 + j * self.sublen[i]
                hi = self.len[i] + m - 1 + (j + 1) * self.sublen[i]
                g[:, i] += _sphere(xx[:, lo:hi])

        g = g / np.maximum(self.sublen[None, :], 1) / self.nk
        shape = np.fliplr(
            np.cumprod(np.column_stack([np.ones((n, 1)), np.cos(pop_dec[:, : m - 1] * np.pi / 2.0)]), axis=1)
        ) * np.column_stack([np.ones((n, 1)), np.sin(pop_dec[:, m - 2 :: -1] * np.pi / 2.0)])

        out["F"] = (1.0 + g + np.column_stack([g[:, 1:], np.zeros(n, dtype=float)])) * (1.0 - shape)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, self.n_obj)
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return 1.0 - r


_CPU = [f"MaF{i}" for i in range(1, 16)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
