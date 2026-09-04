from __future__ import annotations

"""
DAS-CMOP benchmark family converted for local EmoPyLab use.

Reference
---------
Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, and E. Goodman.
Difficulty adjustable and scalable constrained multi-objective test
problem toolkit. Evolutionary Computation, 2020, 28(3): 339-378.
"""

import numpy as np
from core.problem import Problem


def _dascmop_index_from_name(name: str) -> int | None:
    text = str(name).replace("_JAX", "")
    if not text.startswith("DASCMOP"):
        return None
    try:
        return int(text[len("DASCMOP") :])
    except Exception:  # noqa: BLE001
        return None


def _simplex_pf(n: int, m: int) -> np.ndarray:
    n = max(32, int(n))
    m = max(2, int(m))
    if m == 2:
        x = np.linspace(0.0, 1.0, n, dtype=float)
        return np.column_stack([x, 1.0 - x])
    rng = np.random.default_rng(1)
    w = rng.random((n, m))
    den = np.sum(w, axis=1, keepdims=True)
    den[den == 0.0] = 1.0
    return w / den


def _sphere_pf(n: int, m: int) -> np.ndarray:
    r = np.clip(_simplex_pf(n, m), 0.0, None)
    nrm = np.linalg.norm(r, axis=1, keepdims=True)
    nrm[nrm == 0.0] = 1.0
    return r / nrm


class DASCMOP(Problem):
    def __init__(self, n_obj, n_ieq_constr, difficulty_factors, n_var=30, **kwargs):
        super().__init__(
            n_var=int(n_var),
            n_obj=int(n_obj),
            n_ieq_constr=int(n_ieq_constr),
            vtype=float,
            xl=0.0,
            xu=1.0,
            **kwargs,
        )
        self.eta, self.zeta, self.gamma = [float(v) for v in difficulty_factors]

    def g1(self, X):
        contrib = (X[:, self.n_obj - 1 :] - np.sin(0.5 * np.pi * X[:, 0:1])) ** 2
        return contrib.sum(axis=1)[:, None]

    def g2(self, X):
        z = X[:, self.n_obj - 1 :] - 0.5
        contrib = z**2 - np.cos(20 * np.pi * z)
        return (self.n_var - self.n_obj + 1) + contrib.sum(axis=1)[:, None]

    def g3(self, X):
        j = np.arange(self.n_obj - 1, self.n_var) + 1
        contrib = (X[:, self.n_obj - 1 :] - np.cos(0.25 * j / self.n_var * np.pi * (X[:, 0:1] + X[:, 1:2]))) ** 2
        return contrib.sum(axis=1)[:, None]

    def _calc_pareto_front(self, n_pareto_points: int = 300):
        # Objective-space PF approximation. Constraints are intentionally ignored here
        # to keep PF-based metrics usable (avoid NaN) when exact feasible PF generation
        # is not implemented.
        n = max(64, int(n_pareto_points))
        idx = _dascmop_index_from_name(self.__class__.__name__)
        if idx is None:
            return None

        if int(self.n_obj) == 3:
            if idx == 7:
                return _simplex_pf(n, 3)
            if idx in {8, 9}:
                return _sphere_pf(n, 3)
            return None

        x = np.linspace(0.0, 1.0, n, dtype=float)
        if idx in {1, 4}:
            return np.column_stack([x, 1.0 - x**2])
        if idx in {2, 5}:
            return np.column_stack([x, 1.0 - np.sqrt(x)])
        if idx in {3, 6}:
            return np.column_stack([x, 1.0 - np.sqrt(x) + 0.5 * np.abs(np.sin(5.0 * np.pi * x))])
        return None


class DASCMOP1(DASCMOP):
    def __init__(self, n_var=30, difficulty_factors=(0.0, 0.5, 0.5), **kwargs):
        super().__init__(2, 11, difficulty_factors, n_var=n_var, **kwargs)

    def constraints(self, X, f0, f1, g):
        a = 20.0
        b = 2.0 * self.eta - 1.0
        d = 0.5 if self.zeta != 0 else 0.0
        e = d - np.log(self.zeta) if self.zeta > 0 else 1e30
        r = 0.5 * self.gamma

        p_k = np.array([[0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0]])
        q_k = np.array([[1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5]])
        a_k2 = 0.3
        b_k2 = 1.2
        theta_k = -0.25 * np.pi

        c = np.zeros((X.shape[0], 2 + p_k.shape[1]))
        c[:, 0] = np.sin(a * np.pi * X[:, 0]) - b
        if self.zeta == 1.0:
            c[:, 1:2] = 1e-4 - np.abs(e - g)
        else:
            c[:, 1:2] = (e - g) * (g - d)

        c[:, 2:] = (
            ((f0 - p_k) * np.cos(theta_k) - (f1 - q_k) * np.sin(theta_k)) ** 2 / a_k2
            + ((f0 - p_k) * np.sin(theta_k) + (f1 - q_k) * np.cos(theta_k)) ** 2 / b_k2
            - r
        )
        return -1.0 * c

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g1(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - X[:, 0:1] ** 2 + g
        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP2(DASCMOP1):
    def __init__(self, n_var=30, difficulty_factors=(0.0, 0.5, 0.5), **kwargs):
        super().__init__(n_var=n_var, difficulty_factors=difficulty_factors, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g1(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - np.sqrt(X[:, 0:1]) + g
        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP3(DASCMOP1):
    def __init__(self, n_var=30, difficulty_factors=(0.5, 0.5, 0.5), **kwargs):
        super().__init__(n_var=n_var, difficulty_factors=difficulty_factors, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g1(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - np.sqrt(X[:, 0:1]) + 0.5 * np.abs(np.sin(5 * np.pi * X[:, 0:1])) + g
        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP4(DASCMOP1):
    def __init__(self, n_var=30, difficulty_factors=(0.5, 0.5, 0.5), **kwargs):
        super().__init__(n_var=n_var, difficulty_factors=difficulty_factors, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g2(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - X[:, 0:1] ** 2 + g
        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP5(DASCMOP1):
    def __init__(self, n_var=30, difficulty_factors=(0.5, 0.5, 0.5), **kwargs):
        super().__init__(n_var=n_var, difficulty_factors=difficulty_factors, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g2(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - np.sqrt(X[:, 0:1]) + g
        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP6(DASCMOP1):
    def __init__(self, n_var=30, difficulty_factors=(0.5, 0.5, 0.5), **kwargs):
        super().__init__(n_var=n_var, difficulty_factors=difficulty_factors, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g2(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - np.sqrt(X[:, 0:1]) + 0.5 * np.abs(np.sin(5 * np.pi * X[:, 0:1])) + g
        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP7(DASCMOP):
    def __init__(self, n_var=30, difficulty_factors=(0.5, 0.5, 0.5), **kwargs):
        super().__init__(3, 7, difficulty_factors, n_var=n_var, **kwargs)

    def constraints(self, X, f0, f1, f2, g):
        a = 20.0
        b = 2.0 * self.eta - 1.0
        d = 0.5 if self.zeta != 0 else 0.0
        e = d - np.log(self.zeta) if self.zeta > 0 else 1e30
        r = 0.5 * self.gamma

        x_k = np.array([[1.0, 0.0, 0.0, 1.0 / np.sqrt(3.0)]])
        y_k = np.array([[0.0, 1.0, 0.0, 1.0 / np.sqrt(3.0)]])
        z_k = np.array([[0.0, 0.0, 1.0, 1.0 / np.sqrt(3.0)]])

        c = np.zeros((X.shape[0], 3 + x_k.shape[1]))
        c[:, 0] = np.sin(a * np.pi * X[:, 0]) - b
        c[:, 1] = np.cos(a * np.pi * X[:, 1]) - b
        if self.zeta == 1.0:
            c[:, 2:3] = 1e-4 - np.abs(e - g)
        else:
            c[:, 2:3] = (e - g) * (g - d)

        c[:, 3:] = (f0 - x_k) ** 2 + (f1 - y_k) ** 2 + (f2 - z_k) ** 2 - r**2
        return -1.0 * c

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g2(X)
        f0 = X[:, 0:1] * X[:, 1:2] + g
        f1 = X[:, 1:2] * (1.0 - X[:, 0:1]) + g
        f2 = 1.0 - X[:, 1:2] + g
        out["F"] = np.column_stack([f0, f1, f2])
        out["G"] = self.constraints(X, f0, f1, f2, g)


class DASCMOP8(DASCMOP7):
    def __init__(self, n_var=30, difficulty_factors=(0.5, 0.5, 0.5), **kwargs):
        super().__init__(n_var=n_var, difficulty_factors=difficulty_factors, **kwargs)

    @staticmethod
    def objectives(X, g):
        f0 = np.cos(0.5 * np.pi * X[:, 0:1]) * np.cos(0.5 * np.pi * X[:, 1:2]) + g
        f1 = np.cos(0.5 * np.pi * X[:, 0:1]) * np.sin(0.5 * np.pi * X[:, 1:2]) + g
        f2 = np.sin(0.5 * np.pi * X[:, 0:1]) + g
        return np.column_stack([f0, f1, f2])

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g2(X)
        F = self.objectives(X, g)
        out["F"] = F
        out["G"] = self.constraints(X, F[:, 0:1], F[:, 1:2], F[:, 2:3], g)


class DASCMOP9(DASCMOP8):
    def __init__(self, n_var=30, difficulty_factors=(0.5, 0.5, 0.5), **kwargs):
        super().__init__(n_var=n_var, difficulty_factors=difficulty_factors, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        g = self.g3(X)
        F = self.objectives(X, g)
        out["F"] = F
        out["G"] = self.constraints(X, F[:, 0:1], F[:, 1:2], F[:, 2:3], g)


_CPU = [f"DASCMOP{i}" for i in range(1, 10)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
