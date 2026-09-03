from __future__ import annotations

"""
ZCAT benchmark family.

Reference
---------
S. Zapotecas-Martinez, C. A. Coello Coello, H. E. Aguirre, and K. Tanaka.
Challenging test problems for multi- and many-objective optimization.
Swarm and Evolutionary Computation, 2023, 81: 101350.
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


def _g0(y: np.ndarray, theta: float) -> np.ndarray:
    _ = theta
    return np.zeros(y.shape[0], dtype=float) + 0.2210


def _g1(y: np.ndarray, theta: float) -> np.ndarray:
    return np.mean(np.sin(1.5 * np.pi * y + theta), axis=1) / 2.0 + 0.5


def _g2(y: np.ndarray, theta: float) -> np.ndarray:
    return np.mean(y**2 * np.sin(4.5 * np.pi * y + theta), axis=1) / 2.0 + 0.5


def _g3(y: np.ndarray, theta: float) -> np.ndarray:
    return np.mean(np.cos(np.pi * y + theta) ** 2, axis=1)


def _g4(y: np.ndarray, theta: float) -> np.ndarray:
    return np.mean(y * np.cos(4.0 * np.pi * np.mean(y, axis=1, keepdims=True) + theta), axis=1) / 2.0 + 0.5


def _g5(y: np.ndarray, theta: float) -> np.ndarray:
    return np.mean(np.sin(2.0 * np.pi * y - 1.0 + theta) ** 3, axis=1) / 2.0 + 0.5


def _g6(y: np.ndarray, theta: float) -> np.ndarray:
    num = -10.0 * np.exp(-2.0 / 5.0 * np.sqrt(np.mean(y**2, axis=1))) + 10.0 + np.e - np.exp(np.mean(np.cos(11.0 * np.pi * y + theta) ** 3, axis=1))
    den = -10.0 * np.exp(-2.0 / 5.0) - np.exp(-1.0) + 10.0 + np.e
    return num / den


def _g7(y: np.ndarray, theta: float) -> np.ndarray:
    num = np.mean(y, axis=1) + np.exp(np.sin(7.0 * np.pi * np.mean(y, axis=1) - np.pi / 2.0 + theta)) - np.exp(-1.0)
    den = 1.0 + np.exp(1.0) - np.exp(-1.0)
    return num / den


def _g8(y: np.ndarray, theta: float) -> np.ndarray:
    return np.mean(np.abs(np.sin(2.5 * np.pi * (y - 0.5) + theta)), axis=1)


def _g9(y: np.ndarray, theta: float) -> np.ndarray:
    return np.mean(y, axis=1) / 2.0 - np.mean(np.abs(np.sin(2.5 * np.pi * y - np.pi / 2.0 + theta)), axis=1) / 2.0 + 0.5


def _g10(y: np.ndarray, theta: float) -> np.ndarray:
    return np.sum(np.sin((4.0 * y - 2.0) * np.pi + theta) ** 3, axis=1) / (2.0 * (y.shape[1] ** 3)) + 0.5


_G_FUNCS = {
    "g0": _g0,
    "g1": _g1,
    "g2": _g2,
    "g3": _g3,
    "g4": _g4,
    "g5": _g5,
    "g6": _g6,
    "g7": _g7,
    "g8": _g8,
    "g9": _g9,
    "g10": _g10,
}


class _BaseZCAT(Problem):
    _VARIANT = 1
    _COMPLICATED_G = "g4"

    def __init__(self, n_obj: int = 3, n_var: int | None = None, complicatedPS: int = 0, bias: int = 0, imbalance: int = 0, **kwargs):
        self.complicatedPS = int(complicatedPS)
        self.bias = int(bias)
        self.imbalance = int(imbalance)

        n_obj = max(2, int(n_obj))
        d_min = 2 * n_obj if self._VARIANT <= 10 else 2 * n_obj - 1
        if n_var is None:
            n_var = d_min
        n_var = max(int(n_var), d_min)

        lower = -np.arange(1, n_var + 1, dtype=float) / 2.0
        upper = np.arange(1, n_var + 1, dtype=float) / 2.0

        super().__init__(n_var=n_var, n_obj=n_obj, xl=lower, xu=upper, vtype=float, **kwargs)

    def _coef_cumsum(self) -> np.ndarray:
        m = self.n_obj
        den = np.concatenate([[m - 1], np.arange(m - 1, 0, -1, dtype=float)])
        return (np.arange(1, m + 1, dtype=float) ** 2) / den

    def _scale(self) -> np.ndarray:
        return np.arange(1, self.n_obj + 1, dtype=float) ** 2

    def _A(self, y: np.ndarray) -> np.ndarray:
        n, m = y.shape[0], self.n_obj
        head = y[:, : m - 1]
        y1 = y[:, [0]]
        ones = np.ones((n, 1), dtype=float)
        scale = self._scale()[None, :]

        if self._VARIANT == 1:
            a = np.fliplr(np.cumprod(np.sin(head * np.pi / 2.0), axis=1))
            b = np.concatenate([ones, np.cos(y[:, m - 2 : 0 : -1] * np.pi / 2.0)], axis=1)
            a = np.concatenate([a * b, 1.0 - np.sin(y1 * np.pi / 2.0)], axis=1)
            return scale * a

        if self._VARIANT == 2:
            a = np.fliplr(np.cumprod(np.concatenate([ones, 1.0 - np.cos(head * np.pi / 2.0)], axis=1), axis=1))
            b = np.concatenate([ones, 1.0 - np.sin(y[:, m - 2 :: -1] * np.pi / 2.0)], axis=1)
            return scale * a * b

        if self._VARIANT == 3:
            a = np.fliplr(np.cumsum(np.concatenate([np.zeros((n, 1)), head], axis=1), axis=1))
            b = np.concatenate([np.zeros((n, 1)), 1.0 - y[:, m - 2 :: -1]], axis=1)
            return self._coef_cumsum()[None, :] * (a + b)

        if self._VARIANT == 4:
            a = np.concatenate([head, 1.0 - np.mean(head, axis=1, keepdims=True)], axis=1)
            return scale * a

        if self._VARIANT == 5:
            tail = (np.exp(np.mean(1.0 - head, axis=1, keepdims=True)) ** 8 - 1.0) / (np.exp(1.0) ** 8 - 1.0)
            return scale * np.concatenate([head, tail], axis=1)

        if self._VARIANT == 6:
            mean_head = np.mean(head, axis=1, keepdims=True)
            tail = (
                (1.0 / (1.0 + np.exp(80.0 * mean_head - 40.0)) - 0.05 * mean_head - 1.0 / (1.0 + np.exp(40.0)) + 0.05)
                / (1.0 / (1.0 + np.exp(-40.0)) - 1.0 / (1.0 + np.exp(40.0)) + 0.05)
            )
            return scale * np.concatenate([head, tail], axis=1)

        if self._VARIANT == 7:
            tail = np.mean((0.5 - head) ** 5, axis=1, keepdims=True) / (2.0 * (0.5**5)) + 0.5
            return scale * np.concatenate([head, tail], axis=1)

        if self._VARIANT == 8:
            a = np.fliplr(np.cumprod(np.concatenate([ones, 1.0 - np.sin(head * np.pi / 2.0)], axis=1), axis=1))
            b = np.concatenate([ones, 1.0 - np.cos(y[:, m - 2 :: -1] * np.pi / 2.0)], axis=1)
            return scale * (1.0 - a * b)

        if self._VARIANT == 9:
            a = np.fliplr(np.cumsum(np.concatenate([np.zeros((n, 1)), np.sin(np.pi / 2.0 * head)], axis=1), axis=1))
            b = np.concatenate([np.zeros((n, 1)), np.cos(np.pi / 2.0 * y[:, m - 2 :: -1])], axis=1)
            return self._coef_cumsum()[None, :] * (a + b)

        if self._VARIANT == 10:
            tail = (1.0 / 0.02 - 1.0 / (np.mean(1.0 - head, axis=1, keepdims=True) + 0.02)) / (1.0 / 0.02 - 1.0 / 1.02)
            return scale * np.concatenate([head, tail], axis=1)

        if self._VARIANT == 11:
            a = np.fliplr(np.cumsum(head, axis=1))
            b = np.concatenate([np.zeros((n, 1)), 1.0 - y[:, m - 2 : 0 : -1]], axis=1)
            last = (np.cos(7.0 * np.pi * y1) + 2.0 * y1 + 16.0 * (1.0 - y1) - 1.0) / 16.0
            a = np.concatenate([a + b, last], axis=1)
            return self._coef_cumsum()[None, :] * a

        if self._VARIANT == 12:
            a = 1.0 - np.fliplr(np.cumprod(1.0 - head, axis=1)) * np.concatenate([ones, y[:, m - 2 : 0 : -1]], axis=1)
            last = (np.cos(5.0 * np.pi * y1) + 2.0 * y1 + 12.0 * (1.0 - y1) - 1.0) / 12.0
            return scale * np.concatenate([a, last], axis=1)

        if self._VARIANT == 13:
            a = np.fliplr(np.cumsum(np.sin(np.pi / 2.0 * head), axis=1))
            b = np.concatenate([np.zeros((n, 1)), np.cos(np.pi / 2.0 * y[:, m - 2 : 0 : -1])], axis=1)
            last = (np.cos(5.0 * np.pi * y1) + 2.0 * y1 + 12.0 * (1.0 - y1) - 1.0) / 12.0
            c = np.concatenate([a + b, last], axis=1)
            c = 1.0 - c / np.concatenate([[m - 1], np.arange(m - 1, 0, -1, dtype=float)])[None, :]
            return scale * c

        if self._VARIANT == 14:
            s = np.sin(np.pi / 2.0 * y1)
            parts = [s**2]
            if m - 3 >= 1:
                exps = 2.0 + np.arange(1, m - 2, dtype=float) / (m - 2)
                parts.append(s**exps[None, :])
            if m > 2:
                parts.append((1.0 + np.sin(3.0 * np.pi * y1 - np.pi / 2.0)) / 2.0)
            parts.append(np.cos(np.pi / 2.0 * y1))
            return scale * np.concatenate(parts, axis=1)

        if self._VARIANT == 15:
            exps = 1.0 + np.arange(0, m - 1, dtype=float) / (4.0 * m)
            left = y1**exps[None, :]
            last = (np.cos(5.0 * np.pi * y1) + 2.0 * y1 + 12.0 * (1.0 - y1) - 1.0) / 12.0
            return scale * np.concatenate([left, last], axis=1)

        if self._VARIANT == 16:
            s = np.sin(np.pi / 2.0 * y1)
            parts = [s]
            if m - 3 >= 1:
                exps = 1.0 + np.arange(1, m - 2, dtype=float) / (m - 2)
                parts.append(s**exps[None, :])
            if m > 2:
                parts.append((1.0 + np.sin(5.0 * np.pi * y1 - np.pi / 2.0)) / 2.0)
            parts.append((np.cos(9.0 * np.pi * y1) + 2.0 * y1 + 20.0 * (1.0 - y1) - 1.0) / 20.0)
            return scale * np.concatenate(parts, axis=1)

        if self._VARIANT == 17:
            t = np.all(head <= 0.5, axis=1)
            tail = (np.exp(np.mean(1.0 - head, axis=1, keepdims=True)) ** 8 - 1.0) / (np.exp(1.0) ** 8 - 1.0)
            a = np.concatenate([head, tail], axis=1)
            if np.any(t):
                y0 = y[t, 0][:, None]
                a[t, : m - 1] = np.repeat(y0, m - 1, axis=1)
                a[t, m - 1] = ((np.exp(1.0 - y[t, 0]) ** 8 - 1.0) / (np.exp(1.0) ** 8 - 1.0))
            return scale * a

        if self._VARIANT == 18:
            t = np.all((head <= 0.4) | (head >= 0.6), axis=1)
            tail = np.sum((0.5 - head) ** 5, axis=1, keepdims=True) / (2.0 * (m - 1) * (0.5**5)) + 0.5
            a = np.concatenate([head, tail], axis=1)
            if np.any(t):
                y0 = y[t, 0][:, None]
                a[t, : m - 1] = np.repeat(y0, m - 1, axis=1)
                a[t, m - 1] = ((0.5 - y[t, 0]) ** 5 + 0.5**5) / (2.0 * (0.5**5))
            return scale * a

        if self._VARIANT == 19:
            cond = (y[:, 0] <= 0.2) | ((y[:, 0] >= 0.4) & (y[:, 0] <= 0.6))
            t = cond | (m == 2)
            mean_head = np.mean(head, axis=1, keepdims=True)
            tail = 1.0 - mean_head - np.cos(10.0 * np.pi * mean_head + np.pi / 2.0) / (10.0 * np.pi)
            a = np.concatenate([head, tail], axis=1)
            if np.any(t):
                y0 = y[t, 0][:, None]
                a[t, : m - 1] = np.repeat(y0, m - 1, axis=1)
                a[t, m - 1] = 1.0 - y[t, 0] - np.cos(10.0 * np.pi * y[t, 0] + np.pi / 2.0) / (10.0 * np.pi)
            return scale * a

        if self._VARIANT == 20:
            cond = ((y[:, 0] >= 0.1) & (y[:, 0] <= 0.4)) | ((y[:, 0] >= 0.6) & (y[:, 0] <= 0.9))
            t = cond | (m == 2)
            tail = np.sum((0.5 - head) ** 5, axis=1, keepdims=True) / (2.0 * (m - 1) * (0.5**5)) + 0.5
            a = np.concatenate([head, tail], axis=1)
            if np.any(t):
                y0 = y[t, 0][:, None]
                a[t, : m - 1] = np.repeat(y0, m - 1, axis=1)
                a[t, m - 1] = ((0.5 - y[t, 0]) ** 5 + 0.5**5) / (2.0 * (0.5**5))
            return scale * a

        raise RuntimeError(f"Unsupported ZCAT variant: {self._VARIANT}")

    def _evaluate(self, x, out, *args, **kwargs):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.clip(arr, self.xl, self.xu)
        y = (arr - self.xl) / (self.xu - self.xl)

        a = self._A(y)
        base = y[:, : self.n_obj - 1]
        g_name = self._COMPLICATED_G if self.complicatedPS else "g0"
        g_func = _G_FUNCS[g_name]

        z = np.zeros_like(y)
        for j in range(self.n_obj - 1, self.n_var):
            theta = 2.0 * np.pi * (j - (self.n_obj - 1)) / self.n_var
            z[:, j] = y[:, j] - g_func(base, theta)

        w = np.abs(z) ** 0.05 if self.bias else np.abs(z)

        b = np.zeros((arr.shape[0], self.n_obj), dtype=float)
        for i in range(1, self.n_obj + 1):
            cols = np.arange(self.n_obj + i - 2, self.n_var, self.n_obj, dtype=int)
            wi = w[:, cols]
            if wi.shape[1] == 0:
                wi = np.zeros((arr.shape[0], 1), dtype=float)
            if self.imbalance:
                term1 = np.exp(np.max(np.sqrt(np.abs(wi)), axis=1))
                term2 = np.exp(np.mean(0.5 * (np.cos(9.0 * np.pi * wi) + 1.0), axis=1))
                b[:, i - 1] = i**2 * 10.0 / (2.0 * np.e - 2.0) * (term1 - term2 - 1.0 + np.e)
            else:
                b[:, i - 1] = i**2 * 10.0 * np.mean(wi**2, axis=1)

        out["F"] = a + b

    def _calc_pareto_front(self, n_pareto_points=200):
        n = max(20, int(n_pareto_points))
        head = _uniform_box(n, self.n_obj - 1)
        y = np.zeros((head.shape[0], self.n_var), dtype=float)
        y[:, : self.n_obj - 1] = head

        base = y[:, : self.n_obj - 1]
        g_name = self._COMPLICATED_G if self.complicatedPS else "g0"
        g_func = _G_FUNCS[g_name]
        for j in range(self.n_obj - 1, self.n_var):
            theta = 2.0 * np.pi * (j - (self.n_obj - 1)) / self.n_var
            y[:, j] = np.clip(g_func(base, theta), 0.0, 1.0)

        x = self.xl + y * (self.xu - self.xl)
        out = {}
        self._evaluate(x, out)
        f = np.asarray(out["F"], dtype=float)
        mask = _nondominated_mask(f)
        return f[mask]


class ZCAT1(_BaseZCAT):
    _VARIANT = 1
    _COMPLICATED_G = "g4"


class ZCAT2(_BaseZCAT):
    _VARIANT = 2
    _COMPLICATED_G = "g5"


class ZCAT3(_BaseZCAT):
    _VARIANT = 3
    _COMPLICATED_G = "g2"


class ZCAT4(_BaseZCAT):
    _VARIANT = 4
    _COMPLICATED_G = "g7"


class ZCAT5(_BaseZCAT):
    _VARIANT = 5
    _COMPLICATED_G = "g9"


class ZCAT6(_BaseZCAT):
    _VARIANT = 6
    _COMPLICATED_G = "g4"


class ZCAT7(_BaseZCAT):
    _VARIANT = 7
    _COMPLICATED_G = "g5"


class ZCAT8(_BaseZCAT):
    _VARIANT = 8
    _COMPLICATED_G = "g2"


class ZCAT9(_BaseZCAT):
    _VARIANT = 9
    _COMPLICATED_G = "g9"


class ZCAT10(_BaseZCAT):
    _VARIANT = 10
    _COMPLICATED_G = "g7"


class ZCAT11(_BaseZCAT):
    _VARIANT = 11
    _COMPLICATED_G = "g3"


class ZCAT12(_BaseZCAT):
    _VARIANT = 12
    _COMPLICATED_G = "g10"


class ZCAT13(_BaseZCAT):
    _VARIANT = 13
    _COMPLICATED_G = "g1"


class ZCAT14(_BaseZCAT):
    _VARIANT = 14
    _COMPLICATED_G = "g6"


class ZCAT15(_BaseZCAT):
    _VARIANT = 15
    _COMPLICATED_G = "g8"


class ZCAT16(_BaseZCAT):
    _VARIANT = 16
    _COMPLICATED_G = "g10"


class ZCAT17(_BaseZCAT):
    _VARIANT = 17
    _COMPLICATED_G = "g1"


class ZCAT18(_BaseZCAT):
    _VARIANT = 18
    _COMPLICATED_G = "g8"


class ZCAT19(_BaseZCAT):
    _VARIANT = 19
    _COMPLICATED_G = "g6"


class ZCAT20(_BaseZCAT):
    _VARIANT = 20
    _COMPLICATED_G = "g3"


_CPU = [f"ZCAT{i}" for i in range(1, 21)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
