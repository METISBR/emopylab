from __future__ import annotations

"""
IMOP benchmark family.

Reference
---------
Y. Tian, R. Cheng, X. Zhang, M. Li, and Y. Jin.
Diversity assessment of multi-objective evolutionary algorithms:
Performance metric and benchmark problems.
IEEE Computational Intelligence Magazine, 2019, 14(3): 61-74.
"""

import numpy as np
from core.problem import Problem
from util.nds.non_dominated_sorting import NonDominatedSorting


class _BaseIMOP(Problem):
    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)

    @staticmethod
    def _nd_front(f):
        idx = NonDominatedSorting().do(f, only_non_dominated_front=True)
        return f[idx]


class IMOP1(_BaseIMOP):
    def __init__(self, n_var: int | None = None, n_obj: int | None = None, a1: float = 0.05, K: int = 5, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        self.a1 = float(a1)
        self.K = int(K)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        n = max(2, int(n_pareto_points) // 2)
        x = np.linspace(0.5**4, 1.0, n)
        a = (1.0 - x**0.25) ** 4
        f1 = np.concatenate([a[::-1], x])
        f2 = np.concatenate([x[::-1], a])
        return np.column_stack([f1, f2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = min(self.K, self.n_var)
        y1 = np.mean(x[:, :k], axis=1) ** self.a1
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1) if k < self.n_var else np.zeros(x.shape[0])

        f1 = g + np.cos(y1 * np.pi / 2.0) ** 8
        f2 = g + np.sin(y1 * np.pi / 2.0) ** 8
        out["F"] = np.column_stack([f1, f2])


class IMOP2(_BaseIMOP):
    def __init__(self, n_var: int | None = None, n_obj: int | None = None, a1: float = 0.05, K: int = 5, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        self.a1 = float(a1)
        self.K = int(K)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        n = max(2, int(n_pareto_points) // 2)
        x = np.linspace(0.0, 0.5**0.25, n)
        a = (1.0 - x**4) ** 0.25
        f1 = np.concatenate([x, a[::-1]])
        f2 = np.concatenate([a, x[::-1]])
        return np.column_stack([f1, f2])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = min(self.K, self.n_var)
        y1 = np.mean(x[:, :k], axis=1) ** self.a1
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1) if k < self.n_var else np.zeros(x.shape[0])

        f1 = g + np.cos(y1 * np.pi / 2.0) ** 0.5
        f2 = g + np.sin(y1 * np.pi / 2.0) ** 0.5
        out["F"] = np.column_stack([f1, f2])


class IMOP3(_BaseIMOP):
    def __init__(self, n_var: int | None = None, n_obj: int | None = None, a1: float = 0.05, K: int = 5, **kwargs):
        n_obj = 2
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        self.a1 = float(a1)
        self.K = int(K)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=200):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        f = np.column_stack([1.0 + np.cos(x * np.pi * 10.0) / 5.0 - x, x])
        return self._nd_front(f)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = min(self.K, self.n_var)
        y1 = np.mean(x[:, :k], axis=1) ** self.a1
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1) if k < self.n_var else np.zeros(x.shape[0])

        f1 = g + (1.0 + np.cos(y1 * np.pi * 10.0) / 5.0 - y1)
        f2 = g + y1
        out["F"] = np.column_stack([f1, f2])


class IMOP4(_BaseIMOP):
    def __init__(self, n_var: int | None = None, n_obj: int | None = None, a1: float = 0.05, K: int = 5, **kwargs):
        n_obj = 3
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        self.a1 = float(a1)
        self.K = int(K)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, x + np.sin(10.0 * np.pi * x) / 10.0, 1.0 - x])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = min(self.K, self.n_var)
        y1 = np.mean(x[:, :k], axis=1) ** self.a1
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1) if k < self.n_var else np.zeros(x.shape[0])

        s = 1.0 + g
        f1 = s * y1
        f2 = s * (y1 + np.sin(10.0 * np.pi * y1) / 10.0)
        f3 = s * (1.0 - y1)
        out["F"] = np.column_stack([f1, f2, f3])


class IMOP5(_BaseIMOP):
    def __init__(self, n_var: int | None = None, n_obj: int | None = None, a1: float = 0.05, a2: float = 10.0, K: int = 5, **kwargs):
        n_obj = 3
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.K = int(K)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=400):
        m = int(np.ceil(np.sqrt(max(8.0, n_pareto_points / 8.0 * 1.3))))
        xx, yy = np.meshgrid(np.linspace(0.0, 1.0, m), np.linspace(0.0, 1.0, m))
        pts = np.column_stack([xx.ravel(), yy.ravel()]) - 0.5
        pts = 0.2 * pts[np.sum(pts**2, axis=1) <= 0.25]

        centers = np.column_stack([0.4 * np.cos(np.arange(1, 9) * np.pi / 4.0), 0.4 * np.sin(np.arange(1, 9) * np.pi / 4.0)])
        blocks = [pts + c[None, :] for c in centers]
        r = np.vstack(blocks)
        return np.column_stack([r, 0.5 - np.sum(r, axis=1)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = min(self.K, self.n_var)

        odd = np.arange(0, k, 2)
        even = np.arange(1, k, 2)
        y1 = np.mean(x[:, odd], axis=1) ** self.a1 if odd.size > 0 else np.zeros(x.shape[0])
        y2 = np.mean(x[:, even], axis=1) ** self.a2 if even.size > 0 else np.zeros(x.shape[0])
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1) if k < self.n_var else np.zeros(x.shape[0])

        f1 = 0.4 * np.cos(np.pi * np.ceil(y1 * 8.0) / 4.0) + 0.1 * y2 * np.cos(16.0 * np.pi * y1)
        f2 = 0.4 * np.sin(np.pi * np.ceil(y1 * 8.0) / 4.0) + 0.1 * y2 * np.sin(16.0 * np.pi * y1)
        f3 = 0.5 - (f1 + f2)
        out["F"] = np.column_stack([f1 + g, f2 + g, f3 + g])


class IMOP6(_BaseIMOP):
    def __init__(self, n_var: int | None = None, n_obj: int | None = None, a1: float = 0.05, a2: float = 10.0, K: int = 5, **kwargs):
        n_obj = 3
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.K = int(K)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=400):
        m = int(np.ceil(np.sqrt(max(4.0, n_pareto_points))))
        xx, yy = np.meshgrid(np.linspace(0.0, 1.0, m), np.linspace(0.0, 1.0, m))
        r = np.column_stack([xx.ravel(), yy.ravel()])
        d = np.maximum(0.0, np.minimum(np.sin(3.0 * np.pi * r[:, 0]) ** 2, np.sin(3.0 * np.pi * r[:, 1]) ** 2) - 0.05)
        jump = np.ceil(d)
        f = np.column_stack([r[:, 0], r[:, 1], 1.0 - 0.5 * (r[:, 0] + r[:, 1])]) + jump[:, None]
        return self._nd_front(f)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = min(self.K, self.n_var)

        odd = np.arange(0, k, 2)
        even = np.arange(1, k, 2)
        y1 = np.mean(x[:, odd], axis=1) ** self.a1 if odd.size > 0 else np.zeros(x.shape[0])
        y2 = np.mean(x[:, even], axis=1) ** self.a2 if even.size > 0 else np.zeros(x.shape[0])
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1) if k < self.n_var else np.zeros(x.shape[0])

        r = np.maximum(0.0, np.minimum(np.sin(3.0 * np.pi * y1) ** 2, np.sin(3.0 * np.pi * y2) ** 2) - 0.05)
        jump = np.ceil(r)

        f1 = (1.0 + g) * y1 + jump
        f2 = (1.0 + g) * y2 + jump
        f3 = (0.5 + g) * (2.0 - y1 - y2) + jump
        out["F"] = np.column_stack([f1, f2, f3])


class IMOP7(_BaseIMOP):
    def __init__(self, n_var: int | None = None, n_obj: int | None = None, a1: float = 0.05, a2: float = 10.0, K: int = 5, **kwargs):
        n_obj = 3
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.K = int(K)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=400):
        rng = np.random.default_rng(1)
        w = rng.random((max(1, int(n_pareto_points)), 3))
        r = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-30)
        d = np.minimum(np.minimum(np.abs(r[:, 0] - r[:, 1]), np.abs(r[:, 1] - r[:, 2])), np.abs(r[:, 2] - r[:, 0]))
        return r[d <= 0.1]

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = min(self.K, self.n_var)

        odd = np.arange(0, k, 2)
        even = np.arange(1, k, 2)
        y1 = np.mean(x[:, odd], axis=1) ** self.a1 if odd.size > 0 else np.zeros(x.shape[0])
        y2 = np.mean(x[:, even], axis=1) ** self.a2 if even.size > 0 else np.zeros(x.shape[0])
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1) if k < self.n_var else np.zeros(x.shape[0])

        s = 1.0 + g
        f1 = s * np.cos(y1 * np.pi / 2.0) * np.cos(y2 * np.pi / 2.0)
        f2 = s * np.cos(y1 * np.pi / 2.0) * np.sin(y2 * np.pi / 2.0)
        f3 = s * np.sin(y1 * np.pi / 2.0)

        r = np.minimum(np.minimum(np.abs(f1 - f2), np.abs(f2 - f3)), np.abs(f3 - f1))
        bump = 10.0 * np.maximum(0.0, r - 0.1)
        out["F"] = np.column_stack([f1 + bump, f2 + bump, f3 + bump])


class IMOP8(_BaseIMOP):
    def __init__(self, n_var: int | None = None, n_obj: int | None = None, a1: float = 0.05, a2: float = 10.0, K: int = 5, **kwargs):
        n_obj = 3
        n_var = int(n_var) if n_var is not None else 10
        kwargs.pop("n_obj", None)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.K = int(K)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=1000):
        m = int(np.ceil(np.sqrt(max(4.0, n_pareto_points))))
        xx, yy = np.meshgrid(np.linspace(0.0, 1.0, m), np.linspace(0.0, 1.0, m))
        f = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            3.0 - xx.ravel() * (1.0 + np.sin(19.0 * np.pi * xx.ravel())) - yy.ravel() * (1.0 + np.sin(19.0 * np.pi * yy.ravel())),
        ])
        return self._nd_front(f)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        k = min(self.K, self.n_var)

        odd = np.arange(0, k, 2)
        even = np.arange(1, k, 2)
        y1 = np.mean(x[:, odd], axis=1) ** self.a1 if odd.size > 0 else np.zeros(x.shape[0])
        y2 = np.mean(x[:, even], axis=1) ** self.a2 if even.size > 0 else np.zeros(x.shape[0])
        g = np.sum((x[:, k:] - 0.5) ** 2, axis=1) if k < self.n_var else np.zeros(x.shape[0])

        f1 = y1
        f2 = y2
        den = 1.0 + g
        t = (f1 / den) * (1.0 + np.sin(19.0 * np.pi * f1)) + (f2 / den) * (1.0 + np.sin(19.0 * np.pi * f2))
        f3 = den * (3.0 - t)
        out["F"] = np.column_stack([f1, f2, f3])


for _name in [f"IMOP{i}" for i in range(1, 9)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"IMOP{i}" for i in range(1, 9)),
    *(f"IMOP{i}_JAX" for i in range(1, 9)),
]
