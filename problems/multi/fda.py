from __future__ import annotations

"""
FDA dynamic benchmark family.

Reference
---------
M. Farina, K. Deb, and P. Amato.
Dynamic multiobjective optimization problems: Test cases, approximations, and applications.
IEEE Transactions on Evolutionary Computation, 2004, 8(5): 425-442.
"""

import numpy as np
from core.problem import Problem


class _BaseFDA(Problem):
    def __init__(self, *, n_var: int, n_obj: int, taut: int = 10, nt: int = 10, n_pop: int = 100, xl=0.0, xu=1.0, **kwargs):
        self.taut = int(taut)
        self.nt = int(nt)
        self.n_pop = int(max(1, n_pop))
        self._n_eval = 0
        super().__init__(n_var=int(n_var), n_obj=int(n_obj), xl=xl, xu=xu, vtype=float, **kwargs)

    def _clip(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.clip(x, self.xl, self.xu)

    def _time(self):
        return np.floor(self._n_eval / (self.n_pop * self.taut)) / self.nt

    def _advance(self, n):
        self._n_eval += int(n)


def _map_cos_sin(x, m):
    n = x.shape[0]
    if m == 1:
        return np.ones((n, 1))
    a = np.fliplr(np.cumprod(np.column_stack([np.ones((n, 1)), np.cos(x[:, : m - 1] * np.pi / 2.0)]), axis=1))
    b = np.column_stack([np.ones((n, 1)), np.sin(x[:, m - 2 :: -1] * np.pi / 2.0)])
    return a * b


class FDA1(_BaseFDA):
    def __init__(self, n_var: int = 10, taut: int = 10, nt: int = 10, n_pop: int = 100, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(max(0, n - 1))))
        xu = np.concatenate(([1.0], np.ones(max(0, n - 1))))
        super().__init__(n_var=n, n_obj=2, taut=taut, nt=nt, n_pop=n_pop, xl=xl, xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, 1.0 - np.sqrt(x)])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        t = self._time()
        f1 = x[:, 0]
        g = 1.0 + np.sum((x[:, 1:] - np.sin(0.5 * np.pi * t)) ** 2, axis=1)
        h = 1.0 - np.sqrt(f1 / np.maximum(g, 1e-30))
        f2 = g * h
        out["F"] = np.column_stack([f1, f2])
        self._advance(x.shape[0])


class FDA2(_BaseFDA):
    def __init__(self, n_var: int = 10, taut: int = 10, nt: int = 10, n_pop: int = 100, **kwargs):
        n = int(np.ceil((int(n_var) - 1) / 2.0) * 2 + 1)
        xl = np.concatenate(([0.0], -np.ones(max(0, n - 1))))
        xu = np.concatenate(([1.0], np.ones(max(0, n - 1))))
        super().__init__(n_var=n, n_obj=2, taut=taut, nt=nt, n_pop=n_pop, xl=xl, xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        t = self._time()
        h = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        expo = h + max(0.0, h - 1.0) ** 2 * (self.n_var - 1) / 2.0
        return np.column_stack([x, 1.0 - x**expo])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        t = self._time()

        f1 = x[:, 0]
        mid = (self.n_var + 1) // 2
        g = 1.0 + np.sum(x[:, 1:mid] ** 2, axis=1)
        h = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        expo = h + np.sum((x[:, mid:] - h) ** 2, axis=1)
        f2 = g * (1.0 - (f1 / np.maximum(g, 1e-30)) ** expo)

        out["F"] = np.column_stack([f1, f2])
        self._advance(x.shape[0])


class FDA3(_BaseFDA):
    def __init__(self, n_var: int = 10, taut: int = 10, nt: int = 10, n_pop: int = 100, **kwargs):
        n = int(n_var)
        xl = np.concatenate(([0.0], -np.ones(max(0, n - 1))))
        xu = np.concatenate(([1.0], np.ones(max(0, n - 1))))
        super().__init__(n_var=n, n_obj=2, taut=taut, nt=nt, n_pop=n_pop, xl=xl, xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        t = self._time()
        g0 = 1.0 + np.abs(np.sin(0.5 * np.pi * t))
        x = np.linspace(0.0, 1.0, int(n_pareto_points))
        return np.column_stack([x, g0 * (1.0 - np.sqrt(x / g0))])

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        t = self._time()

        f1 = x[:, 0] ** (10.0 ** (2.0 * np.sin(0.5 * np.pi * t)))
        g0 = np.abs(np.sin(0.5 * np.pi * t))
        g = 1.0 + g0 + np.sum((x[:, 1:] - g0) ** 2, axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / np.maximum(g, 1e-30)))

        out["F"] = np.column_stack([f1, f2])
        self._advance(x.shape[0])


class FDA4(_BaseFDA):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, taut: int = 10, nt: int = 10, n_pop: int = 100, **kwargs):
        m = int(n_obj)
        n = int(m + 9 if n_var is None else n_var)
        super().__init__(n_var=n, n_obj=m, taut=taut, nt=nt, n_pop=n_pop, xl=0.0, xu=1.0, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        r = np.random.default_rng(1).random((max(1, int(n_pareto_points)), self.n_obj))
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        t = self._time()
        g0 = np.abs(np.sin(0.5 * np.pi * t))
        g = np.sum((x[:, self.n_obj - 1 :] - g0) ** 2, axis=1)

        f = (1.0 + g)[:, None] * _map_cos_sin(x, self.n_obj)
        out["F"] = f
        self._advance(x.shape[0])


class FDA5(_BaseFDA):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, taut: int = 10, nt: int = 10, n_pop: int = 100, **kwargs):
        m = int(n_obj)
        n = int(m + 9 if n_var is None else n_var)
        super().__init__(n_var=n, n_obj=m, taut=taut, nt=nt, n_pop=n_pop, xl=0.0, xu=1.0, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        t = self._time()
        g0 = np.abs(np.sin(0.5 * np.pi * t))
        r = np.random.default_rng(1).random((max(1, int(n_pareto_points)), self.n_obj))
        r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
        return r * (1.0 + g0)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._clip(x)
        t = self._time()
        xt = x.copy()
        xt[:, : self.n_obj - 1] = xt[:, : self.n_obj - 1] ** (1.0 + 100.0 * np.sin(0.5 * np.pi * t) ** 4)

        g0 = np.abs(np.sin(0.5 * np.pi * t))
        g = g0 + np.sum((xt[:, self.n_obj - 1 :] - g0) ** 2, axis=1)

        f = (1.0 + g)[:, None] * _map_cos_sin(xt, self.n_obj)
        out["F"] = f
        self._advance(x.shape[0])


for _name in [f"FDA{i}" for i in range(1, 6)]:
    _cls = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_cls,), {})


__all__ = [
    *(f"FDA{i}" for i in range(1, 6)),
    *(f"FDA{i}_JAX" for i in range(1, 6)),
]
