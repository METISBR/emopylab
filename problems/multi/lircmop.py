from __future__ import annotations

"""
LIR-CMOP benchmark family.

Reference
---------
Z. Fan, W. Li, X. Cai, H. Huang, Y. Fang, Y. You, J. Mo, C. Wei, and E. Goodman.
An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions.
Soft Computing, 2019, 23: 12491-12510.
"""

import numpy as np
from core.problem import Problem


def _uniform_simplex(n: int, m: int, seed: int = 1) -> np.ndarray:
    n = max(1, int(n))
    m = max(1, int(m))
    if m == 1:
        return np.ones((n, 1))
    if m == 2:
        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([x, 1.0 - x])
    rng = np.random.default_rng(seed)
    w = rng.random((n, m))
    return w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-30)


def _ellipse_constraints(f, p, q, a, b, r=0.1, theta=-0.25 * np.pi):
    f1 = f[:, 0]
    f2 = f[:, 1]
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    out = []
    ct = np.cos(theta)
    st = np.sin(theta)
    for pk, qk, ak, bk in zip(p, q, a, b):
        t1 = (f1 - pk) * ct - (f2 - qk) * st
        t2 = (f1 - pk) * st + (f2 - qk) * ct
        out.append(r - (t1**2) / (ak**2) - (t2**2) / (bk**2))
    return np.column_stack(out)


def _constraint_alpha(f, c):
    alpha = 0.25 * np.pi
    f1 = f[:, 0]
    f2 = f[:, 1]
    return c - f1 * np.sin(alpha) - f2 * np.cos(alpha) + np.sin(4.0 * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha)))


def _sum_terms_variant_5_to_12(x):
    n, d = x.shape
    if d <= 1:
        z = np.zeros(n)
        return z, z

    x0 = x[:, 0]
    idx = np.arange(1, d)
    j = idx + 1  # MATLAB index
    ang = 0.5 * j[None, :] / d * np.pi * x0[:, None]

    sub = x[:, 1:]
    odd = (j % 2) == 1
    even = ~odd

    s1 = np.sum((sub[:, odd] - np.sin(ang[:, odd])) ** 2, axis=1) if np.any(odd) else np.zeros(n)
    s2 = np.sum((sub[:, even] - np.cos(ang[:, even])) ** 2, axis=1) if np.any(even) else np.zeros(n)
    return s1, s2


class _BaseLIRCMOP(Problem):
    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)


class LIRCMOP1(_BaseLIRCMOP):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - x**2]) + 0.5

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        x_odd = x[:, 2::2]
        x_even = x[:, 1::2]

        g1 = np.sum((x_odd - np.sin(0.5 * np.pi * x[:, [0]])) ** 2, axis=1)
        g2 = np.sum((x_even - np.cos(0.5 * np.pi * x[:, [0]])) ** 2, axis=1)

        f1 = x[:, 0] + g1
        f2 = 1.0 - x[:, 0] ** 2 + g2
        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.5 - g2) * (0.51 - g2)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2])


class LIRCMOP2(_BaseLIRCMOP):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)]) + 0.5

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        x_odd = x[:, 2::2]
        x_even = x[:, 1::2]

        g1 = np.sum((x_odd - x[:, [0]]) ** 2, axis=1)
        g2 = np.sum((x_even - x[:, [0]]) ** 2, axis=1)

        f1 = x[:, 0] + g1
        f2 = 1.0 - np.sqrt(x[:, 0]) + g2
        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.5 - g2) * (0.51 - g2)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2])


class LIRCMOP3(_BaseLIRCMOP):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=3, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f = np.column_stack([x, 1.0 - x**2])
        f = f[np.sin(20.0 * np.pi * x) >= 0.5]
        return f + 0.5

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        x_odd = x[:, 2::2]
        x_even = x[:, 1::2]

        g1 = np.sum((x_odd - x[:, [0]]) ** 2, axis=1)
        g2 = np.sum((x_even - x[:, [0]]) ** 2, axis=1)

        f1 = x[:, 0] + g1
        f2 = 1.0 - x[:, 0] ** 2 + g2
        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.5 - g2) * (0.51 - g2)
        c3 = 0.5 - np.sin(20.0 * np.pi * x[:, 0])

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3])


class LIRCMOP4(_BaseLIRCMOP):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=3, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f = np.column_stack([x, 1.0 - np.sqrt(x)])
        f = f[np.sin(20.0 * np.pi * x) >= 0.5]
        return f + 0.5

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        x_odd = x[:, 2::2]
        x_even = x[:, 1::2]

        g1 = np.sum((x_odd - x[:, [0]]) ** 2, axis=1)
        g2 = np.sum((x_even - x[:, [0]]) ** 2, axis=1)

        f1 = x[:, 0] + g1
        f2 = 1.0 - np.sqrt(x[:, 0]) + g2
        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.5 - g2) * (0.51 - g2)
        c3 = 0.5 - np.sin(20.0 * np.pi * x[:, 0])

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3])


class _LIRCMOP5to12(_BaseLIRCMOP):
    def _evaluate_base(self, x):
        x = self._clip(x)
        s1, s2 = _sum_terms_variant_5_to_12(x)
        return x, s1, s2


class LIRCMOP5(_LIRCMOP5to12):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _constraints(self, f):
        return _ellipse_constraints(f, [1.6, 2.5], [1.6, 2.5], [2.0, 2.0], [4.0, 8.0])

    def _calc_pareto_front(self, n_pareto_points=300):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f = np.column_stack([x, 1.0 - np.sqrt(x)]) + 0.7057
        return f[np.all(self._constraints(f) <= 0.0, axis=1)]

    def _evaluate(self, x, out, *args, **kwargs):
        x, s1, s2 = self._evaluate_base(x)
        f = np.column_stack([x[:, 0] + 10.0 * s1 + 0.7057, 1.0 - np.sqrt(x[:, 0]) + 10.0 * s2 + 0.7057])
        out["F"] = f
        out["G"] = self._constraints(f)


class LIRCMOP6(_LIRCMOP5to12):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _constraints(self, f):
        return _ellipse_constraints(f, [1.8, 2.8], [1.8, 2.8], [2.0, 2.0], [8.0, 8.0])

    def _calc_pareto_front(self, n_pareto_points=300):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f = np.column_stack([x, 1.0 - x**2]) + 0.7057
        return f[np.all(self._constraints(f) <= 0.0, axis=1)]

    def _evaluate(self, x, out, *args, **kwargs):
        x, s1, s2 = self._evaluate_base(x)
        f = np.column_stack([x[:, 0] + 10.0 * s1 + 0.7057, 1.0 - x[:, 0] ** 2 + 10.0 * s2 + 0.7057])
        out["F"] = f
        out["G"] = self._constraints(f)


class LIRCMOP7(_LIRCMOP5to12):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=3, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _constraints(self, f):
        return _ellipse_constraints(f, [1.2, 2.25, 3.5], [1.2, 2.25, 3.5], [2.0, 2.5, 2.5], [6.0, 12.0, 10.0])

    def _calc_pareto_front(self, n_pareto_points=300):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f = np.column_stack([x, 1.0 - np.sqrt(x)]) + 0.7057

        theta = -0.25 * np.pi
        c1 = 0.1 - ((f[:, 0] - 1.2) * np.cos(theta) - (f[:, 1] - 1.2) * np.sin(theta)) ** 2 / (2.0**2) - ((f[:, 0] - 1.2) * np.sin(theta) + (f[:, 1] - 1.2) * np.cos(theta)) ** 2 / (6.0**2)
        invalid = c1 > 0
        it = 0
        while np.any(invalid) and it < 1000:
            f[invalid, :] = (f[invalid, :] - 0.7057) * 1.001 + 0.7057
            c1 = 0.1 - ((f[:, 0] - 1.2) * np.cos(theta) - (f[:, 1] - 1.2) * np.sin(theta)) ** 2 / (2.0**2) - ((f[:, 0] - 1.2) * np.sin(theta) + (f[:, 1] - 1.2) * np.cos(theta)) ** 2 / (6.0**2)
            invalid = c1 > 0
            it += 1
        return f

    def _evaluate(self, x, out, *args, **kwargs):
        x, s1, s2 = self._evaluate_base(x)
        f = np.column_stack([x[:, 0] + 10.0 * s1 + 0.7057, 1.0 - np.sqrt(x[:, 0]) + 10.0 * s2 + 0.7057])
        out["F"] = f
        out["G"] = self._constraints(f)


class LIRCMOP8(_LIRCMOP5to12):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=3, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _constraints(self, f):
        return _ellipse_constraints(f, [1.2, 2.25, 3.5], [1.2, 2.25, 3.5], [2.0, 2.5, 2.5], [6.0, 12.0, 10.0])

    def _calc_pareto_front(self, n_pareto_points=300):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        # Keep the MATLAB GetOptimum expression (uses sqrt here).
        f = np.column_stack([x, 1.0 - np.sqrt(x)]) + 0.7057

        theta = -0.25 * np.pi
        c1 = 0.1 - ((f[:, 0] - 1.2) * np.cos(theta) - (f[:, 1] - 1.2) * np.sin(theta)) ** 2 / (2.0**2) - ((f[:, 0] - 1.2) * np.sin(theta) + (f[:, 1] - 1.2) * np.cos(theta)) ** 2 / (6.0**2)
        invalid = c1 > 0
        it = 0
        while np.any(invalid) and it < 1000:
            f[invalid, :] = (f[invalid, :] - 0.7057) * 1.001 + 0.7057
            c1 = 0.1 - ((f[:, 0] - 1.2) * np.cos(theta) - (f[:, 1] - 1.2) * np.sin(theta)) ** 2 / (2.0**2) - ((f[:, 0] - 1.2) * np.sin(theta) + (f[:, 1] - 1.2) * np.cos(theta)) ** 2 / (6.0**2)
            invalid = c1 > 0
            it += 1
        return f

    def _evaluate(self, x, out, *args, **kwargs):
        x, s1, s2 = self._evaluate_base(x)
        f = np.column_stack([x[:, 0] + 10.0 * s1 + 0.7057, 1.0 - x[:, 0] ** 2 + 10.0 * s2 + 0.7057])
        out["F"] = f
        out["G"] = self._constraints(f)


class LIRCMOP9(_LIRCMOP5to12):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _constraints(self, f):
        c1 = _ellipse_constraints(f, [1.4], [1.4], [1.5], [6.0])[:, 0]
        c2 = _constraint_alpha(f, 2.0)
        return np.column_stack([c1, c2])

    def _calc_pareto_front(self, n_pareto_points=300):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f = np.column_stack([x, 1.0 - x**2]) * 1.7057
        f = f[np.all(self._constraints(f) <= 0.0, axis=1)]
        return np.vstack([f, np.array([[0.0, 2.182], [1.856, 0.0]])])

    def _evaluate(self, x, out, *args, **kwargs):
        x, s1, s2 = self._evaluate_base(x)
        f = np.column_stack([1.7057 * x[:, 0] * (10.0 * s1 + 1.0), 1.7057 * (1.0 - x[:, 0] ** 2) * (10.0 * s2 + 1.0)])
        out["F"] = f
        out["G"] = self._constraints(f)


class LIRCMOP10(_LIRCMOP5to12):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _constraints(self, f):
        c1 = _ellipse_constraints(f, [1.1], [1.2], [2.0], [4.0])[:, 0]
        c2 = _constraint_alpha(f, 1.0)
        return np.column_stack([c1, c2])

    def _calc_pareto_front(self, n_pareto_points=300):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f = np.column_stack([x, 1.0 - np.sqrt(x)]) * 1.7057
        f = f[np.all(self._constraints(f) <= 0.0, axis=1)]
        return np.vstack([f, np.array([[1.747, 0.0]])])

    def _evaluate(self, x, out, *args, **kwargs):
        x, s1, s2 = self._evaluate_base(x)
        f = np.column_stack([1.7057 * x[:, 0] * (10.0 * s1 + 1.0), 1.7057 * (1.0 - np.sqrt(x[:, 0])) * (10.0 * s2 + 1.0)])
        out["F"] = f
        out["G"] = self._constraints(f)


class LIRCMOP11(_LIRCMOP5to12):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _constraints(self, f):
        c1 = _ellipse_constraints(f, [1.2], [1.2], [1.5], [5.0])[:, 0]
        c2 = _constraint_alpha(f, 2.1)
        return np.column_stack([c1, c2])

    def _calc_pareto_front(self, n_pareto_points=100):
        _ = n_pareto_points
        return np.array(
            [
                [1.3965, 0.1591],
                [1.0430, 0.5127],
                [0.6894, 0.8662],
                [0.3359, 1.2198],
                [0.0106, 1.6016],
                [0.0, 2.1910],
                [1.8730, 0.0],
            ],
            dtype=float,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x, s1, s2 = self._evaluate_base(x)
        f = np.column_stack([1.7057 * x[:, 0] * (10.0 * s1 + 1.0), 1.7057 * (1.0 - np.sqrt(x[:, 0])) * (10.0 * s2 + 1.0)])
        out["F"] = f
        out["G"] = self._constraints(f)


class LIRCMOP12(_LIRCMOP5to12):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _constraints(self, f):
        c1 = _ellipse_constraints(f, [1.6], [1.6], [1.5], [6.0])[:, 0]
        c2 = _constraint_alpha(f, 2.5)
        return np.column_stack([c1, c2])

    def _calc_pareto_front(self, n_pareto_points=100):
        _ = n_pareto_points
        return np.array(
            [
                [1.6794, 0.4419],
                [1.3258, 0.7955],
                [0.9723, 1.1490],
                [2.0320, 0.0990],
                [0.6187, 1.5026],
                [0.2652, 1.8562],
                [0.0, 2.2580],
                [2.5690, 0.0],
            ],
            dtype=float,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x, s1, s2 = self._evaluate_base(x)
        f = np.column_stack([1.7057 * x[:, 0] * (10.0 * s1 + 1.0), 1.7057 * (1.0 - x[:, 0] ** 2) * (10.0 * s2 + 1.0)])
        out["F"] = f
        out["G"] = self._constraints(f)


class LIRCMOP13(_BaseLIRCMOP):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=3, n_ieq_constr=2, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, 3)
        r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return 1.7057 * r

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        s = np.sum(10.0 * (x[:, 2:] - 0.5) ** 2, axis=1) if self.n_var > 2 else np.zeros(x.shape[0])
        scale = 1.7057 + s

        f1 = scale * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
        f2 = scale * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
        f3 = scale * np.sin(0.5 * np.pi * x[:, 0])
        gx = f1**2 + f2**2 + f3**2

        c1 = (gx - 9.0) * (4.0 - gx)
        c2 = (gx - 3.61) * (3.24 - gx)

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([c1, c2])


class LIRCMOP14(_BaseLIRCMOP):
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=3, n_ieq_constr=3, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, 3)
        r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return np.sqrt(3.0625) * r

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        s = np.sum(10.0 * (x[:, 2:] - 0.5) ** 2, axis=1) if self.n_var > 2 else np.zeros(x.shape[0])
        scale = 1.7057 + s

        f1 = scale * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
        f2 = scale * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
        f3 = scale * np.sin(0.5 * np.pi * x[:, 0])
        gx = f1**2 + f2**2 + f3**2

        c1 = (gx - 9.0) * (4.0 - gx)
        c2 = (gx - 3.61) * (3.24 - gx)
        c3 = (gx - 3.0625) * (2.56 - gx)

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([c1, c2, c3])


for _name in [f"LIRCMOP{i}" for i in range(1, 15)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"LIRCMOP{i}" for i in range(1, 15)),
    *(f"LIRCMOP{i}_JAX" for i in range(1, 15)),
]
