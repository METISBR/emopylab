from __future__ import annotations

"""
DOC constrained benchmark family.

Reference
---------
Z. Liu and Y. Wang.
Handling constrained multiobjective optimization problems with constraints
in both the decision and objective spaces.
IEEE Transactions on Evolutionary Computation, 2019, 23(5): 870-884.
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


class _BaseDOC(Problem):
    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)


class DOC1(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.array([0.0, 78.0, 33.0, 27.0, 27.0, 27.0], dtype=float)
        xu = np.array([1.0, 102.0, 45.0, 45.0, 45.0, 45.0], dtype=float)
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=7, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        r = _uniform_simplex(n_pareto_points, 2)
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = 5.3578547 * x[:, 3] ** 2 + 0.8356891 * x[:, 1] * x[:, 5] + 37.293239 * x[:, 1] - 40792.141 + 30665.5386717834 + 1.0

        f1 = x[:, 0]
        f2 = g * (1.0 - np.sqrt(f1) / np.maximum(g, 1e-30))

        c1 = np.maximum(-(f1**2 + f2**2 - 1.0), 0.0)
        c2 = 85.334407 + 0.0056858 * x[:, 2] * x[:, 5] + 0.0006262 * x[:, 1] * x[:, 4] - 0.0022053 * x[:, 3] * x[:, 5] - 92.0
        c3 = -85.334407 - 0.0056858 * x[:, 2] * x[:, 5] - 0.0006262 * x[:, 1] * x[:, 4] + 0.0022053 * x[:, 3] * x[:, 5]
        c4 = 80.51249 + 0.0071317 * x[:, 2] * x[:, 5] + 0.0029955 * x[:, 1] * x[:, 2] + 0.0021813 * x[:, 3] ** 2 - 110.0
        c5 = -80.51249 - 0.0071317 * x[:, 2] * x[:, 5] - 0.0029955 * x[:, 1] * x[:, 2] - 0.0021813 * x[:, 3] ** 2 + 90.0
        c6 = 9.300961 + 0.0047026 * x[:, 3] * x[:, 5] + 0.0012547 * x[:, 1] * x[:, 3] + 0.0019085 * x[:, 3] * x[:, 4] - 25.0
        c7 = -9.300961 - 0.0047026 * x[:, 3] * x[:, 5] - 0.0012547 * x[:, 1] * x[:, 3] - 0.0019085 * x[:, 3] * x[:, 4] + 20.0

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6, c7])


class DOC2(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.concatenate(([0.0], np.zeros(15)))
        xu = np.concatenate(([1.0], 10.0 * np.ones(15)))
        super().__init__(n_var=16, n_obj=2, n_ieq_constr=7, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        r1 = np.linspace(0.0, 1.0, int(n_pareto_points))
        r2 = 1.0 - np.sqrt(r1)
        mask = ~((r1 < 0.05) | ((r1 > 0.2202) & (r1 < 0.3830)) | ((r1 > 0.6247) & (r1 < 0.7440)))
        return np.column_stack([r1[mask], r2[mask]])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        n = x.shape[0]

        b = np.array([-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0], dtype=float)
        cmat = np.array(
            [
                [30.0, -20.0, -10.0, 32.0, -10.0],
                [-20.0, 39.0, -6.0, -31.0, 32.0],
                [-10.0, -6.0, 10.0, -6.0, -10.0],
                [32.0, -31.0, -6.0, 39.0, -20.0],
                [-10.0, 32.0, -10.0, -20.0, 30.0],
            ],
            dtype=float,
        )
        d = np.array([4.0, 8.0, 10.0, 6.0, 2.0], dtype=float)

        xv = x[:, 11:16]
        q = xv @ cmat
        g_temp = np.sum(q * xv, axis=1) + 2.0 * np.sum(d[None, :] * xv**3, axis=1) - np.sum(b[None, :] * x[:, 1:11], axis=1)
        g = (g_temp - 32.6555929502) + 1.0

        f1 = x[:, 0]
        f2 = g * (1.0 - np.power(np.maximum(f1, 0.0), 1.0 / 3.0) / np.maximum(g, 1e-30))

        c1 = np.maximum(-(np.sqrt(np.maximum(f1, 0.0)) + f2 - 1.0), 0.0)
        d1_1 = np.maximum((f1 - 1.0 / 8.0) ** 2 + (f2 - 1.0 + np.sqrt(1.0 / 8.0)) ** 2 - 0.15**2, 0.0)
        d1_2 = np.maximum((f1 - 1.0 / 2.0) ** 2 + (f2 - 1.0 + np.sqrt(1.0 / 2.0)) ** 2 - 0.15**2, 0.0)
        d1_3 = np.maximum((f1 - 7.0 / 8.0) ** 2 + (f2 - 1.0 + np.sqrt(7.0 / 8.0)) ** 2 - 0.15**2, 0.0)
        c2 = np.min(np.column_stack([d1_1, d1_2, d1_3]), axis=1)

        amat = np.array(
            [
                [-16.0, 2.0, 0.0, 1.0, 0.0],
                [0.0, -2.0, 0.0, 0.4, 2.0],
                [-3.5, 0.0, 2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0, -4.0, -1.0],
                [0.0, -9.0, -2.0, 1.0, -2.8],
                [2.0, 0.0, -4.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -2.0, -3.0, -2.0, -1.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        e = np.array([-15.0, -27.0, -36.0, -18.0, -12.0], dtype=float)
        ax = x[:, 1:11] @ amat
        s = xv @ cmat

        c3 = -2.0 * s[:, 0] - 3.0 * d[0] * x[:, 11] ** 2 - e[0] + ax[:, 0]
        c4 = -2.0 * s[:, 1] - 3.0 * d[1] * x[:, 12] ** 2 - e[1] + ax[:, 1]
        c5 = -2.0 * s[:, 2] - 3.0 * d[2] * x[:, 13] ** 2 - e[2] + ax[:, 2]
        c6 = -2.0 * s[:, 3] - 3.0 * d[3] * x[:, 14] ** 2 - e[3] + ax[:, 3]
        c7 = -2.0 * s[:, 4] - 3.0 * d[4] * x[:, 15] ** 2 - e[4] + ax[:, 4]

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6, c7])


class DOC3(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01], dtype=float)
        xu = np.array([1.0, 1.0, 300.0, 100.0, 200.0, 100.0, 1.0, 100.0, 200.0, 0.03], dtype=float)
        super().__init__(n_var=10, n_obj=2, n_ieq_constr=10, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        r = _uniform_simplex(n_pareto_points, 2)
        r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        x = r[:, 0]
        mask = ~(((x > 0.3403) & (x < 0.4782)) | ((x > 0.6553) & (x < 0.7553)) | ((x > 0.8782) & (x < 0.9403)))
        return r[mask]

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g_temp = -9.0 * x[:, 5] - 15.0 * x[:, 8] + 6.0 * x[:, 1] + 16.0 * x[:, 2] + 10.0 * (x[:, 6] + x[:, 7])
        g = (g_temp + 400.0551) + 1.0

        f1 = x[:, 0]
        f2 = g * (1.0 - f1 / np.maximum(g, 1e-30))

        s2 = np.sqrt(2.0)
        c1 = np.maximum(-(f1**2 + f2**2 - 1.0), 0.0)
        c2 = np.maximum(-(np.abs((-f1 + f2 - 0.5) / s2) - 0.1 / s2), 0.0)
        c3 = np.maximum(-(np.abs((-f1 + f2 - 0.0) / s2) - 0.1 / s2), 0.0)
        c4 = np.maximum(-(np.abs((-f1 + f2 + 0.5) / s2) - 0.1 / s2), 0.0)

        c5 = x[:, 9] * x[:, 3] + 0.02 * x[:, 6] - 0.025 * x[:, 5]
        c6 = x[:, 9] * x[:, 4] + 0.02 * x[:, 7] - 0.015 * x[:, 8]
        c7 = np.abs(x[:, 1] + x[:, 2] - x[:, 3] - x[:, 4]) - 1e-4
        c8 = np.abs(0.03 * x[:, 1] + 0.01 * x[:, 2] - x[:, 9] * (x[:, 3] + x[:, 4])) - 1e-4
        c9 = np.abs(x[:, 3] + x[:, 6] - x[:, 5]) - 1e-4
        c10 = np.abs(x[:, 4] + x[:, 7] - x[:, 8]) - 1e-4

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])


class DOC4(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.array([0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], dtype=float)
        xu = np.array([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
        super().__init__(n_var=8, n_obj=2, n_ieq_constr=6, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=21):
        r1 = np.arange(0, 21, dtype=float) / 20.0
        r2 = 1.0 - r1
        return np.column_stack([r1, r2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g_temp = (
            (x[:, 1] - 10.0) ** 2
            + 5.0 * (x[:, 2] - 12.0) ** 2
            + x[:, 3] ** 4
            + 3.0 * (x[:, 4] - 11.0) ** 2
            + 10.0 * x[:, 5] ** 6
            + 7.0 * x[:, 6] ** 2
            + x[:, 7] ** 4
            - 4.0 * x[:, 6] * x[:, 7]
            - 10.0 * x[:, 6]
            - 8.0 * x[:, 7]
        )
        g = g_temp - 680.6300573745 + 1.0

        f1 = x[:, 0]
        f2 = g * (1.0 - np.sqrt(f1) / np.maximum(g, 1e-30))

        c1 = np.maximum(-(f1 + f2 - 1.0), 0.0)
        c2 = np.maximum(-(f1 + f2 - 1.0 - np.abs(np.sin(10.0 * np.pi * (f1 - f2 + 1.0)))), 0.0)
        c3 = -127.0 + 2.0 * x[:, 1] ** 2 + 3.0 * x[:, 2] ** 4 + x[:, 3] + 4.0 * x[:, 4] ** 2 + 5.0 * x[:, 5]
        c4 = -282.0 + 7.0 * x[:, 1] + 3.0 * x[:, 2] + 10.0 * x[:, 3] ** 2 + x[:, 4] - x[:, 5]
        c5 = -196.0 + 23.0 * x[:, 1] + x[:, 2] ** 2 + 6.0 * x[:, 6] ** 2 - 8.0 * x[:, 7]
        c6 = 4.0 * x[:, 1] ** 2 + x[:, 2] ** 2 - 3.0 * x[:, 1] * x[:, 2] + 2.0 * x[:, 3] ** 2 + 5.0 * x[:, 6] - 11.0 * x[:, 7]

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6])


class DOC5(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.array([0.0, 0.0, 0.0, 0.0, 100.0, 6.3, 5.9, 4.5], dtype=float)
        xu = np.array([1.0, 1000.0, 40.0, 40.0, 300.0, 6.7, 6.4, 6.25], dtype=float)
        super().__init__(n_var=8, n_obj=2, n_ieq_constr=9, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=14):
        r1 = np.concatenate([np.arange(0, 9, dtype=float), np.arange(16, 21, dtype=float)]) / 20.0
        r2 = 1.0 - r1
        return np.column_stack([r1, r2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = (x[:, 1] - 193.724510070035) + 1.0

        f1 = x[:, 0]
        f2 = g * (1.0 - np.sqrt(f1) / np.maximum(g, 1e-30))

        c1 = np.maximum(-(f1 + f2 - 1.0), 0.0)
        c2 = np.maximum(-(f1 + f2 - 1.0 - np.abs(np.sin(10.0 * np.pi * (f1 - f2 + 1.0)))), 0.0)
        c3 = np.maximum((f1 - 0.8) * (f2 - 0.6), 0.0)

        c4 = -x[:, 1] + 35.0 * x[:, 2] ** 0.6 + 35.0 * x[:, 3] ** 0.6
        c5 = np.abs(-300.0 * x[:, 3] + 7500.0 * x[:, 5] - 7500.0 * x[:, 6] - 25.0 * x[:, 4] * x[:, 5] + 25.0 * x[:, 4] * x[:, 6] + x[:, 3] * x[:, 4]) - 1e-4
        c6 = np.abs(100.0 * x[:, 2] + 155.365 * x[:, 4] + 2500.0 * x[:, 7] - x[:, 2] * x[:, 4] - 25.0 * x[:, 4] * x[:, 7] - 15536.5) - 1e-4
        c7 = np.abs(-x[:, 5] + np.log(-x[:, 4] + 900.0)) - 1e-4
        c8 = np.abs(-x[:, 6] + np.log(x[:, 4] + 300.0)) - 1e-4
        c9 = np.abs(-x[:, 7] + np.log(-2.0 * x[:, 4] + 700.0)) - 1e-4

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6, c7, c8, c9])


class DOC6(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.concatenate(([0.0], -10.0 * np.ones(10)))
        xu = np.concatenate(([1.0], 10.0 * np.ones(10)))
        super().__init__(n_var=11, n_obj=2, n_ieq_constr=10, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        n = max(2, int(n_pareto_points))
        r1 = np.concatenate([np.linspace(0.0, 0.5, n), np.arange(11, 21, dtype=float) / 20.0])
        r2 = 1.0 - r1
        return np.column_stack([r1, r2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g_temp = (
            x[:, 1] ** 2
            + x[:, 2] ** 2
            + x[:, 1] * x[:, 2]
            - 14.0 * x[:, 1]
            - 16.0 * x[:, 2]
            + (x[:, 3] - 10.0) ** 2
            + 4.0 * (x[:, 4] - 5.0) ** 2
            + (x[:, 5] - 3.0) ** 2
            + 2.0 * (x[:, 6] - 1.0) ** 2
            + 5.0 * x[:, 7] ** 2
            + 7.0 * (x[:, 8] - 11.0) ** 2
            + 2.0 * (x[:, 9] - 10.0) ** 2
            + (x[:, 10] - 7.0) ** 2
            + 45.0
        )
        g = g_temp - 24.3062090681 + 1.0

        f1 = x[:, 0]
        f2 = g * (1.0 - np.sqrt(f1) / np.maximum(g, 1e-30))

        c1 = np.maximum(-(f1 + f2 - 1.0), 0.0)
        c2 = np.maximum(-(f1 - 0.5) * (f1 + f2 - 1.0 - np.abs(np.sin(10.0 * np.pi * (f1 - f2 + 1.0)))), 0.0)

        c3 = -105.0 + 4.0 * x[:, 1] + 5.0 * x[:, 2] - 3.0 * x[:, 7] + 9.0 * x[:, 8]
        c4 = 10.0 * x[:, 1] - 8.0 * x[:, 2] - 17.0 * x[:, 7] + 2.0 * x[:, 8]
        c5 = -8.0 * x[:, 1] + 2.0 * x[:, 2] + 5.0 * x[:, 9] - 2.0 * x[:, 10] - 12.0
        c6 = 3.0 * (x[:, 1] - 2.0) ** 2 + 4.0 * (x[:, 2] - 3.0) ** 2 + 2.0 * x[:, 3] ** 2 - 7.0 * x[:, 4] - 120.0
        c7 = 5.0 * x[:, 1] ** 2 + 8.0 * x[:, 2] + (x[:, 3] - 6.0) ** 2 - 2.0 * x[:, 4] - 40.0
        c8 = x[:, 1] ** 2 + 2.0 * (x[:, 2] - 2.0) ** 2 - 2.0 * x[:, 1] * x[:, 2] + 14.0 * x[:, 5] - 6.0 * x[:, 6]
        c9 = 0.5 * (x[:, 1] - 8.0) ** 2 + 2.0 * (x[:, 2] - 4.0) ** 2 + 3.0 * x[:, 5] ** 2 - x[:, 6] - 30.0
        c10 = -3.0 * x[:, 1] + 6.0 * x[:, 2] + 12.0 * (x[:, 9] - 8.0) ** 2 - 7.0 * x[:, 10]

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])


class DOC7(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.concatenate(([0.0], np.zeros(10)))
        xu = np.concatenate(([1.0], 10.0 * np.ones(10)))
        super().__init__(n_var=11, n_obj=2, n_ieq_constr=6, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        n = max(2, int(n_pareto_points))
        r1 = np.concatenate([np.linspace(0.0, 0.45, n), np.arange(11, 21, dtype=float) / 20.0])
        r2 = 1.0 - r1
        return np.column_stack([r1, r2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        cvec = np.array([-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179], dtype=float)
        xs = x[:, 1:11]
        denom = 1e-30 + np.sum(xs, axis=1, keepdims=True)
        g_temp = np.sum(xs * (cvec[None, :] + np.log(1e-30 + xs / denom)), axis=1)
        g = g_temp + 47.7648884595 + 1.0

        f1 = x[:, 0]
        f2 = g * (1.0 - np.sqrt(f1) / np.maximum(g, 1e-30))

        c1 = np.maximum(-(f1 + f2 - 1.0), 0.0)
        c2 = np.maximum(-(f1 - 0.5) * (f1 + f2 - 1.0 - np.abs(np.sin(10.0 * np.pi * (f1 - f2 + 1.0)))), 0.0)
        c3 = np.maximum(-(np.abs(-f1 + f2) / np.sqrt(2.0) - 0.1 / np.sqrt(2.0)), 0.0)

        c4 = np.abs(x[:, 1] + 2.0 * x[:, 2] + 2.0 * x[:, 3] + x[:, 6] + x[:, 10] - 2.0) - 1e-4
        c5 = np.abs(x[:, 4] + 2.0 * x[:, 5] + x[:, 6] + x[:, 7] - 1.0) - 1e-4
        c6 = np.abs(x[:, 3] + x[:, 7] + x[:, 8] + 2.0 * x[:, 9] + x[:, 10] - 1.0) - 1e-4

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6])


class DOC8(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.array([0.0, 0.0, 500.0, 1000.0, 5000.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=float)
        xu = np.array([1.0, 1.0, 1000.0, 2000.0, 6000.0, 500.0, 500.0, 500.0, 500.0, 500.0], dtype=float)
        super().__init__(n_var=10, n_obj=3, n_ieq_constr=7, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        r = _uniform_simplex(n_pareto_points, 3)
        return r[(r[:, 2] <= 0.4) | (r[:, 2] >= 0.6)]

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g = (x[:, 2] + x[:, 3] + x[:, 4] - 7049.2480205286) + 1.0

        f1 = (x[:, 0] * x[:, 1]) * g
        f2 = (x[:, 0] * (1.0 - x[:, 1])) * g
        f3 = (1.0 - x[:, 0]) * g

        c1 = np.maximum(-((f3 - 0.4) * (f3 - 0.6)), 0.0)
        c2 = -1.0 + 0.0025 * (x[:, 5] + x[:, 7])
        c3 = -1.0 + 0.0025 * (x[:, 6] + x[:, 8] - x[:, 5])
        c4 = -1.0 + 0.01 * (x[:, 9] - x[:, 6])
        c5 = -x[:, 2] * x[:, 7] + 833.33252 * x[:, 5] + 100.0 * x[:, 2] - 83333.333
        c6 = -x[:, 3] * x[:, 8] + 1250.0 * x[:, 6] + x[:, 3] * x[:, 5] - 1250.0 * x[:, 5]
        c7 = -x[:, 4] * x[:, 9] + 1250000.0 + x[:, 4] * x[:, 6] - 2500.0 * x[:, 6]

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6, c7])


class DOC9(_BaseDOC):
    def __init__(self, **kwargs):
        xl = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=float)
        xu = np.array([1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
        super().__init__(n_var=11, n_obj=3, n_ieq_constr=14, xl=xl, xu=xu, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        r = _uniform_simplex(n_pareto_points, 2)
        r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return np.column_stack([r, np.zeros((r.shape[0], 1))])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        g_temp = -0.5 * (x[:, 2] * x[:, 5] - x[:, 3] * x[:, 4] + x[:, 4] * x[:, 10] - x[:, 6] * x[:, 10] + x[:, 6] * x[:, 9] - x[:, 7] * x[:, 8])
        g = g_temp + 0.8660254038 + 1.0

        f1 = np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1]) * g
        f2 = np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1]) * g
        f3 = np.sin(0.5 * np.pi * x[:, 0]) * g

        c1 = np.maximum(-(f1**2 + f2**2 - 1.0), 0.0)
        c2 = x[:, 4] ** 2 + x[:, 5] ** 2 - 1.0
        c3 = x[:, 10] ** 2 - 1.0
        c4 = x[:, 6] ** 2 + x[:, 7] ** 2 - 1.0
        c5 = x[:, 2] ** 2 + (x[:, 3] - x[:, 10]) ** 2 - 1.0
        c6 = (x[:, 2] - x[:, 6]) ** 2 + (x[:, 3] - x[:, 7]) ** 2 - 1.0
        c7 = (x[:, 2] - x[:, 8]) ** 2 + (x[:, 3] - x[:, 9]) ** 2 - 1.0
        c8 = (x[:, 4] - x[:, 6]) ** 2 + (x[:, 5] - x[:, 7]) ** 2 - 1.0
        c9 = (x[:, 4] - x[:, 8]) ** 2 + (x[:, 5] - x[:, 9]) ** 2 - 1.0
        c10 = x[:, 8] ** 2 + (x[:, 9] - x[:, 10]) ** 2 - 1.0
        c11 = x[:, 3] * x[:, 4] - x[:, 2] * x[:, 5]
        c12 = -x[:, 4] * x[:, 10]
        c13 = x[:, 6] * x[:, 10]
        c14 = x[:, 7] * x[:, 8] - x[:, 6] * x[:, 9]

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14])


for _name in [f"DOC{i}" for i in range(1, 10)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"DOC{i}" for i in range(1, 10)),
    *(f"DOC{i}_JAX" for i in range(1, 10)),
]
