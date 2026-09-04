from __future__ import annotations

"""
CF constrained benchmark family (CEC 2009).

Reference
---------
Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, and S. Tiwari.
Multiobjective optimization test instances for the CEC 2009 special
session and competition. University of Essex, Working Report CES-487, 2009.
"""

import numpy as np
from core.problem import Problem


class _BaseCF(Problem):
    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)

    def _calc_pareto_front(self, n_pareto_points: int = 300):
        # Practical PF approximations in objective space.
        # For constrained variants, this intentionally ignores feasibility filtering for
        # some families to avoid NaN in PF-based metrics when no exact PF generator exists.
        n = max(32, int(n_pareto_points))
        idx = _cf_index_from_name(self.__class__.__name__)
        if idx is None:
            return None

        if int(getattr(self, "n_obj", 2)) == 3:
            return _cf_sphere_pf(n)

        x = np.linspace(0.0, 1.0, n, dtype=float)
        if idx in {1, 4, 5}:
            return np.column_stack([x, 1.0 - x])
        if idx == 2:
            return np.column_stack([x, 1.0 - np.sqrt(x)])
        if idx == 3:
            return np.column_stack([x, 1.0 - x**2])
        if idx in {6, 7}:
            return np.column_stack([x, (1.0 - x) ** 2])
        return None


def _cf_index_from_name(name: str) -> int | None:
    text = str(name).replace("_JAX", "")
    try:
        if text.startswith("CF"):
            return int(text[2:])
    except Exception:  # noqa: BLE001
        return None
    return None


def _cf_sphere_pf(n: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    r = rng.random((max(2, int(n)), 3))
    nrm = np.linalg.norm(r, axis=1, keepdims=True)
    nrm[nrm == 0.0] = 1.0
    return r / nrm


class CF1(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=2, n_ieq_constr=1, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(2, d, 2)
        j2 = np.arange(1, d, 2)

        i1 = (j1 + 1)[None, :]
        i2 = (j2 + 1)[None, :]
        y1 = x[:, j1] - x[:, [0]] ** (0.5 * (1 + 3 * (i1 - 2) / (d - 2)))
        y2 = x[:, j2] - x[:, [0]] ** (0.5 * (1 + 3 * (i2 - 2) / (d - 2)))

        f1 = x[:, 0] + 2 * np.mean(y1**2, axis=1)
        f2 = 1 - x[:, 0] + 2 * np.mean(y2**2, axis=1)
        g = 1 - f1 - f2 + np.abs(np.sin(10 * np.pi * (f1 - f2 + 1)))
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g[:, None]


class CF2(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(n - 1)))
        xu = np.ones(n)
        super().__init__(n_var=n, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(2, d, 2)
        j2 = np.arange(1, d, 2)

        i1 = (j1 + 1)[None, :]
        i2 = (j2 + 1)[None, :]
        y1 = x[:, j1] - np.sin(6 * np.pi * x[:, [0]] + i1 * np.pi / d)
        y2 = x[:, j2] - np.cos(6 * np.pi * x[:, [0]] + i2 * np.pi / d)

        f1 = x[:, 0] + 2 * np.mean(y1**2, axis=1)
        f2 = 1 - np.sqrt(x[:, 0]) + 2 * np.mean(y2**2, axis=1)
        t = f2 + np.sqrt(f1) - np.sin(2 * np.pi * (np.sqrt(f1) - f2 + 1)) - 1
        g = -t / (1 + np.exp(4 * np.abs(t)))
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g[:, None]


class CF3(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -2 * np.ones(n - 1)))
        xu = np.concatenate(([1.0], 2 * np.ones(n - 1)))
        super().__init__(n_var=n, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(2, d, 2)
        j2 = np.arange(1, d, 2)

        idx = np.arange(1, d + 1)[None, :]
        y = x - np.sin(6 * np.pi * x[:, [0]] + idx * np.pi / d)

        i1 = (j1 + 1)[None, :]
        i2 = (j2 + 1)[None, :]
        t1 = 4 * np.sum(y[:, j1] ** 2, axis=1) - 2 * np.prod(np.cos(20 * y[:, j1] * np.pi / np.sqrt(i1)), axis=1) + 2
        t2 = 4 * np.sum(y[:, j2] ** 2, axis=1) - 2 * np.prod(np.cos(20 * y[:, j2] * np.pi / np.sqrt(i2)), axis=1) + 2

        f1 = x[:, 0] + 2 / len(j1) * t1
        f2 = 1 - x[:, 0] ** 2 + 2 / len(j2) * t2
        g = 1 - f2 - f1**2 + np.sin(2 * np.pi * (f1**2 - f2 + 1))
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g[:, None]


class CF4(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -2 * np.ones(n - 1)))
        xu = np.concatenate(([1.0], 2 * np.ones(n - 1)))
        super().__init__(n_var=n, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(2, d, 2)
        j2 = np.arange(1, d, 2)

        idx = np.arange(1, d + 1)[None, :]
        y = x - np.sin(6 * np.pi * x[:, [0]] + idx * np.pi / d)
        h = y**2

        temp = y[:, 1] < (1.5 * (1 - np.sqrt(0.5)))
        h[temp, 1] = np.abs(y[temp, 1])
        h[~temp, 1] = 0.125 + (y[~temp, 1] - 1) ** 2

        f1 = x[:, 0] + np.sum(h[:, j1], axis=1)
        f2 = 1 - x[:, 0] + np.sum(h[:, j2], axis=1)

        t = x[:, 1] - np.sin(6 * np.pi * x[:, 0] + 2 * np.pi / d) - 0.5 * x[:, 0] + 0.25
        g = -t / (1 + np.exp(4 * np.abs(t)))
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g[:, None]


class CF5(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -2 * np.ones(n - 1)))
        xu = np.concatenate(([1.0], 2 * np.ones(n - 1)))
        super().__init__(n_var=n, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(2, d, 2)
        j2 = np.arange(1, d, 2)

        y = np.zeros_like(x)
        i1 = (j1 + 1)[None, :]
        i2 = (j2 + 1)[None, :]
        y[:, j1] = x[:, j1] - 0.8 * x[:, [0]] * np.cos(6 * np.pi * x[:, [0]] + i1 * np.pi / d)
        y[:, j2] = x[:, j2] - 0.8 * x[:, [0]] * np.sin(6 * np.pi * x[:, [0]] + i2 * np.pi / d)

        h = 2 * y**2 - np.cos(4 * np.pi * y) + 1
        temp = y[:, 1] < (1.5 * (1 - np.sqrt(0.5)))
        h[temp, 1] = np.abs(y[temp, 1])
        h[~temp, 1] = 0.125 + (y[~temp, 1] - 1) ** 2

        f1 = x[:, 0] + np.sum(h[:, j1], axis=1)
        f2 = 1 - x[:, 0] + np.sum(h[:, j2], axis=1)
        g = -x[:, 1] + 0.8 * x[:, 0] * np.sin(6 * np.pi * x[:, 0] + 2 * np.pi / d) + 0.5 * x[:, 0] - 0.25

        out["F"] = np.column_stack([f1, f2])
        out["G"] = g[:, None]


class CF6(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -2 * np.ones(n - 1)))
        xu = np.concatenate(([1.0], 2 * np.ones(n - 1)))
        super().__init__(n_var=n, n_obj=2, n_ieq_constr=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(2, d, 2)
        j2 = np.arange(1, d, 2)

        y = np.zeros_like(x)
        i1 = (j1 + 1)[None, :]
        i2 = (j2 + 1)[None, :]
        y[:, j1] = x[:, j1] - 0.8 * x[:, [0]] * np.cos(6 * np.pi * x[:, [0]] + i1 * np.pi / d)
        y[:, j2] = x[:, j2] - 0.8 * x[:, [0]] * np.sin(6 * np.pi * x[:, [0]] + i2 * np.pi / d)

        f1 = x[:, 0] + np.sum(y[:, j1] ** 2, axis=1)
        f2 = (1 - x[:, 0]) ** 2 + np.sum(y[:, j2] ** 2, axis=1)

        a = 0.5 * (1 - x[:, 0]) - (1 - x[:, 0]) ** 2
        b = 0.25 * np.sqrt(1 - x[:, 0]) - 0.5 * (1 - x[:, 0])

        g1 = -x[:, 1] + 0.8 * x[:, 0] * np.sin(6 * np.pi * x[:, 0] + 2 * np.pi / d) + np.sign(a) * np.sqrt(np.abs(a))
        g2 = -x[:, 3] + 0.8 * x[:, 0] * np.sin(6 * np.pi * x[:, 0] + 4 * np.pi / d) + np.sign(b) * np.sqrt(np.abs(b))

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


class CF7(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -2 * np.ones(n - 1)))
        xu = np.concatenate(([1.0], 2 * np.ones(n - 1)))
        super().__init__(n_var=n, n_obj=2, n_ieq_constr=2, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(2, d, 2)
        j2 = np.arange(1, d, 2)

        y = np.zeros_like(x)
        i1 = (j1 + 1)[None, :]
        i2 = (j2 + 1)[None, :]
        y[:, j1] = x[:, j1] - np.cos(6 * np.pi * x[:, [0]] + i1 * np.pi / d)
        y[:, j2] = x[:, j2] - np.sin(6 * np.pi * x[:, [0]] + i2 * np.pi / d)

        h = 2 * y**2 - np.cos(4 * np.pi * y) + 1
        if self.n_var > 3:
            h[:, [1, 3]] = y[:, [1, 3]] ** 2

        f1 = x[:, 0] + np.sum(h[:, j1], axis=1)
        f2 = (1 - x[:, 0]) ** 2 + np.sum(h[:, j2], axis=1)

        a = 0.5 * (1 - x[:, 0]) - (1 - x[:, 0]) ** 2
        b = 0.25 * np.sqrt(1 - x[:, 0]) - 0.5 * (1 - x[:, 0])

        g1 = -x[:, 1] + np.sin(6 * np.pi * x[:, 0] + 2 * np.pi / d) + np.sign(a) * np.sqrt(np.abs(a))
        g2 = -x[:, 3] + np.sin(6 * np.pi * x[:, 0] + 4 * np.pi / d) + np.sign(b) * np.sqrt(np.abs(b))

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


class CF8(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0, 0.0], -4 * np.ones(n - 2)))
        xu = np.concatenate(([1.0, 1.0], 4 * np.ones(n - 2)))
        super().__init__(n_var=n, n_obj=3, n_ieq_constr=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(3, d, 3)
        j2 = np.arange(4, d, 3)
        j3 = np.arange(2, d, 3)

        idx = np.arange(1, d + 1)[None, :]
        y = x - 2 * x[:, [1]] * np.sin(2 * np.pi * x[:, [0]] + idx * np.pi / d)

        f1 = np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, j1] ** 2, axis=1)
        f2 = np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, j2] ** 2, axis=1)
        f3 = np.sin(0.5 * x[:, 0] * np.pi) + 2 * np.mean(y[:, j3] ** 2, axis=1)

        num = f1**2 + f2**2
        den = np.maximum(1e-32, 1 - f3**2)
        g = 1 - num / den + 4 * np.abs(np.sin(2 * np.pi * (num / den + 1)))

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = g[:, None]


class CF9(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0, 0.0], -2 * np.ones(n - 2)))
        xu = np.concatenate(([1.0, 1.0], 2 * np.ones(n - 2)))
        super().__init__(n_var=n, n_obj=3, n_ieq_constr=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(3, d, 3)
        j2 = np.arange(4, d, 3)
        j3 = np.arange(2, d, 3)

        idx = np.arange(1, d + 1)[None, :]
        y = x - 2 * x[:, [1]] * np.sin(2 * np.pi * x[:, [0]] + idx * np.pi / d)

        f1 = np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, j1] ** 2, axis=1)
        f2 = np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, j2] ** 2, axis=1)
        f3 = np.sin(0.5 * x[:, 0] * np.pi) + 2 * np.mean(y[:, j3] ** 2, axis=1)

        num = f1**2 + f2**2
        den = np.maximum(1e-32, 1 - f3**2)
        g = 1 - num / den + 3 * np.sin(2 * np.pi * (num / den + 1))

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = g[:, None]


class CF10(_BaseCF):
    def __init__(self, n_var: int = 10, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0, 0.0], -2 * np.ones(n - 2)))
        xu = np.concatenate(([1.0, 1.0], 2 * np.ones(n - 2)))
        super().__init__(n_var=n, n_obj=3, n_ieq_constr=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        d = self.n_var
        j1 = np.arange(3, d, 3)
        j2 = np.arange(4, d, 3)
        j3 = np.arange(2, d, 3)

        idx = np.arange(1, d + 1)[None, :]
        y = x - 2 * x[:, [1]] * np.sin(2 * np.pi * x[:, [0]] + idx * np.pi / d)

        f1 = np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + 2 * np.mean(4 * y[:, j1] ** 2 - np.cos(8 * np.pi * y[:, j1]) + 1, axis=1)
        f2 = np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + 2 * np.mean(4 * y[:, j2] ** 2 - np.cos(8 * np.pi * y[:, j2]) + 1, axis=1)
        f3 = np.sin(0.5 * x[:, 0] * np.pi) + 2 * np.mean(4 * y[:, j3] ** 2 - np.cos(8 * np.pi * y[:, j3]) + 1, axis=1)

        num = f1**2 + f2**2
        den = np.maximum(1e-32, 1 - f3**2)
        g = 1 - num / den + np.sin(2 * np.pi * (num / den + 1))

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = g[:, None]


_CPU = [f"CF{i}" for i in range(1, 11)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
