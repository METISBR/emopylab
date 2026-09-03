from __future__ import annotations

"""
ZXH_CF constrained benchmark family.

Reference
---------
Y. Zhou, Y. Xiang, and X. He.
Constrained multiobjective optimization: Test problem construction and performance evaluations.
IEEE Transactions on Evolutionary Computation, 2021, 25(1): 172-186.
"""

import numpy as np
from core.problem import Problem


def _uniform_box(n: int, dim: int) -> np.ndarray:
    n = max(2, int(n))
    dim = max(1, int(dim))
    if dim == 1:
        return np.linspace(0.0, 1.0, n)[:, None]
    if dim == 2:
        s = int(np.ceil(np.sqrt(n)))
        x = np.linspace(0.0, 1.0, s)
        xx, yy = np.meshgrid(x, x)
        return np.column_stack([xx.ravel(), yy.ravel()])
    rng = np.random.default_rng(1)
    return rng.random((n, dim))


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
    return keep


def _theta_to_x(theta: np.ndarray, m: int) -> np.ndarray:
    n = theta.shape[0]
    if m == 1:
        return np.ones((n, 1), dtype=float)
    x = np.zeros((n, m), dtype=float)
    for i in range(m):
        if i == 0:
            x[:, i] = np.cos(np.pi / 2.0 * theta[:, 0])
        elif i < m - 1:
            x[:, i] = np.prod(np.sin(np.pi / 2.0 * theta[:, :i]), axis=1) * np.cos(np.pi / 2.0 * theta[:, i])
        else:
            x[:, i] = np.prod(np.sin(np.pi / 2.0 * theta[:, : m - 1]), axis=1)
    return x


_CFG = {
    1: dict(m_default=3, obj="linear", h="sphere", c2=None, theta=None),
    2: dict(m_default=3, obj="concave", h="rosen", c2=0.25, theta=None),
    3: dict(m_default=3, obj="convex", h="ackley", c2=0.5, theta=None),
    4: dict(m_default=3, obj="mixed", h="griewank", c2=0.75, theta=None),
    5: dict(m_default=3, obj="convex", h="rosen", c2=None, theta="max_quarter"),
    6: dict(m_default=3, obj="convex", h="sphere", c2=0.25, theta="max_quarter"),
    7: dict(m_default=3, obj="linear", h="griewank", c2=0.5, theta="max_quarter"),
    8: dict(m_default=3, obj="concave", h="ackley", c2=0.75, theta="max_quarter"),
    9: dict(m_default=3, obj="mixed", h="ackley", c2=None, theta="min_quarter"),
    10: dict(m_default=3, obj="convex", h="griewank", c2=0.25, theta="min_quarter"),
    11: dict(m_default=3, obj="concave", h="sphere", c2=0.5, theta="min_quarter"),
    12: dict(m_default=3, obj="linear", h="rosen", c2=0.75, theta="min_quarter"),
    13: dict(m_default=2, obj="concave", h="griewank", c2=None, theta="piece"),
    14: dict(m_default=2, obj="linear", h="ackley", c2=0.25, theta="piece"),
    15: dict(m_default=2, obj="mixed", h="rosen", c2=0.25, theta="piece"),
    16: dict(m_default=2, obj="convex", h="sphere", c2=0.25, theta="piece"),
}


class _BaseZXHCF(Problem):
    _IDX = 1

    def __init__(self, n_obj: int | None = None, n_var: int | None = None, **kwargs):
        cfg = _CFG[self._IDX]
        if n_obj is None:
            n_obj = int(cfg["m_default"])
        n_obj = max(2, int(n_obj))
        if n_var is None:
            n_var = n_obj + 10
        n_var = max(n_obj + 1, int(n_var))

        self._cfg = cfg
        self.k = self._compute_k(n_obj)
        n_con = self._num_constraints(n_obj)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_con,
            xl=np.zeros(n_var, dtype=float) + 1e-10,
            xu=np.ones(n_var, dtype=float) - 1e-10,
            vtype=float,
            **kwargs,
        )

    @staticmethod
    def _compute_k(m: int) -> int:
        if m <= 3:
            return m - 1
        if m <= 8:
            return int(np.floor(m / 2))
        return 3

    def _num_constraints(self, m: int) -> int:
        cfg = self._cfg
        count = 1
        if cfg["c2"] is not None:
            count += 1
        if cfg["theta"] is not None:
            count += self._compute_k(m)
        return count

    def _h(self, tail: np.ndarray) -> np.ndarray:
        cfg = self._cfg["h"]
        opt = 0.2
        if tail.shape[1] == 0:
            return np.zeros(tail.shape[0], dtype=float)
        if cfg == "sphere":
            return np.sum((tail - opt) ** 2, axis=1)
        if cfg == "rosen":
            a = tail[:, :-1] - opt
            b = tail[:, 1:] - opt
            return np.sum(100.0 * (a**2 - b) ** 2 + a**2, axis=1)
        if cfg == "griewank":
            d = tail.shape[1]
            num = np.sum((tail - opt) ** 2, axis=1)
            den = np.sqrt(np.arange(1, d + 1, dtype=float))[None, :]
            prod = np.prod(np.cos(10.0 * np.pi * (tail - opt) / den), axis=1)
            return 5.0 * (num - prod + 1.0)
        if cfg == "ackley":
            d = tail.shape[1]
            term1 = 20.0 - 20.0 * np.exp(-0.2 * np.sqrt(np.sum((tail - opt) ** 2, axis=1) / d))
            term2 = np.e - np.exp(np.sum(np.cos(2.0 * np.pi * (tail - opt)), axis=1) / d)
            return term1 + term2
        raise RuntimeError(f"Unknown h-function: {cfg}")

    def _theta(self, x_head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sx = np.cumsum(x_head**2, axis=1)[:, ::-1]
        if self.n_obj <= 1:
            return sx, np.zeros((x_head.shape[0], 0), dtype=float)
        theta = 2.0 / np.pi * np.arctan(np.sqrt(sx[:, 1:]) / np.maximum(x_head[:, : self.n_obj - 1], 1e-30))
        return sx, theta

    def _g(self, theta: np.ndarray) -> np.ndarray:
        n = theta.shape[0]
        if self.n_obj <= 1:
            return np.ones((n, 1), dtype=float)
        if self._cfg["obj"] == "linear":
            return np.concatenate([np.ones((n, 1)), np.cumprod(theta, axis=1)], axis=1) * np.concatenate([1.0 - theta, np.ones((n, 1))], axis=1)

        base = np.concatenate([np.ones((n, 1)), np.cumprod(np.sin(np.pi / 2.0 * theta), axis=1)], axis=1) * np.concatenate(
            [np.cos(np.pi / 2.0 * theta), np.ones((n, 1))], axis=1
        )
        if self._cfg["obj"] == "concave":
            return base

        g = 1.0 - base
        if self._cfg["obj"] == "mixed":
            segments = 2.0
            g[:, 0] = theta[:, 0] - np.cos(2.0 * np.pi * segments * theta[:, 0] + np.pi / 2.0) / (2.0 * segments * np.pi)
        return g

    def _theta_constraints(self, theta: np.ndarray) -> list[np.ndarray]:
        style = self._cfg["theta"]
        if style is None:
            return []
        k = self._compute_k(self.n_obj)
        out = []
        for i in range(k):
            th = theta[:, i]
            if style == "max_quarter":
                out.append(np.maximum(0.25 - th, th - 0.75))
            elif style == "min_quarter":
                out.append(np.minimum(th - 0.25, 0.75 - th))
            elif style == "piece":
                out.append(np.minimum(np.minimum(th - 0.1, 0.8 - th), np.maximum(0.4 - th, th - 0.7)))
            else:
                raise RuntimeError(f"Unknown theta-constraint style: {style}")
        return out

    def _evaluate(self, x, out, *args, **kwargs):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.clip(arr, self.xl, self.xu)

        head = arr[:, : self.n_obj]
        tail = arr[:, self.n_obj :]
        sx, theta = self._theta(head)
        h = self._h(tail)
        t = (1.0 - sx[:, 0]) ** 2 + h
        g = self._g(theta)
        f = g * (1.0 + t)[:, None]

        cons = [sx[:, 0] + h - 1.0]
        if self._cfg["c2"] is not None:
            cons.append(-(sx[:, 0] + h - float(self._cfg["c2"])))
        cons.extend(self._theta_constraints(theta))

        out["F"] = f
        out["G"] = np.column_stack(cons)

    def _calc_pareto_front(self, n_pareto_points=200):
        n = max(2000, int(n_pareto_points) * 20)
        theta = _uniform_box(n, self.n_obj - 1)
        head = _theta_to_x(theta, self.n_obj)

        x = np.full((head.shape[0], self.n_var), 0.2, dtype=float)
        x[:, : self.n_obj] = np.clip(head, self.xl[: self.n_obj], self.xu[: self.n_obj])

        out = {}
        self._evaluate(x, out)
        f = np.asarray(out["F"], dtype=float)
        g = np.asarray(out["G"], dtype=float)
        feasible = np.all(g <= 0.0, axis=1)
        if not np.any(feasible):
            feasible = np.ones(f.shape[0], dtype=bool)
        f = f[feasible]
        mask = _nondominated_mask(f)
        return f[mask]


class ZXH_CF1(_BaseZXHCF):
    _IDX = 1


class ZXH_CF2(_BaseZXHCF):
    _IDX = 2


class ZXH_CF3(_BaseZXHCF):
    _IDX = 3


class ZXH_CF4(_BaseZXHCF):
    _IDX = 4


class ZXH_CF5(_BaseZXHCF):
    _IDX = 5


class ZXH_CF6(_BaseZXHCF):
    _IDX = 6


class ZXH_CF7(_BaseZXHCF):
    _IDX = 7


class ZXH_CF8(_BaseZXHCF):
    _IDX = 8


class ZXH_CF9(_BaseZXHCF):
    _IDX = 9


class ZXH_CF10(_BaseZXHCF):
    _IDX = 10


class ZXH_CF11(_BaseZXHCF):
    _IDX = 11


class ZXH_CF12(_BaseZXHCF):
    _IDX = 12


class ZXH_CF13(_BaseZXHCF):
    _IDX = 13


class ZXH_CF14(_BaseZXHCF):
    _IDX = 14


class ZXH_CF15(_BaseZXHCF):
    _IDX = 15


class ZXH_CF16(_BaseZXHCF):
    _IDX = 16


_CPU = [f"ZXH_CF{i}" for i in range(1, 17)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
