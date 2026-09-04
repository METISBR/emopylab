from __future__ import annotations

"""
SMD bilevel test problem family converted for local use in EmoPyLab.

Reference
---------
A. Sinha, P. Malo, and K. Deb.
Test problem construction for single-objective bilevel optimization.
Evolutionary Computation, 2014, 22(3): 439-477.
"""

import numpy as np

from core.problem import Problem


def _as_2d(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _rosenbrock_term(x: np.ndarray) -> np.ndarray:
    if x.shape[1] < 2:
        return np.zeros(x.shape[0], dtype=float)
    return np.sum((x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


class _BaseSMD(Problem):
    _IDX = 1

    def __init__(self, n_var: int = 5, max_felower: int = 500, **kwargs):
        d = max(4, int(n_var))
        self.max_felower = int(max_felower)
        self.du = d // 2
        self.dl = d - self.du
        self.r = self.du // 2
        self.p = self.du - self.r
        self.q = self.dl - self.r
        self.s = 0

        xl, xu = self._bounds(d)
        n_ieq = self._n_ieq_constr()
        super().__init__(n_var=d, n_obj=2, n_ieq_constr=n_ieq, xl=xl, xu=xu, vtype=float, **kwargs)

    def _n_ieq_constr(self) -> int:
        if self._IDX in {1, 2, 3, 4, 5, 6, 7, 8}:
            return 0
        if self._IDX == 9:
            return 2
        if self._IDX == 10:
            return self.du + self.q
        if self._IDX == 11:
            return self.r + 1
        if self._IDX == 12:
            return self.p + 2 * self.r + self.q + 1
        return 0

    def _bounds(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        p, q, r = self.p, self.q, self.r
        if self._IDX in {1, 3}:
            xl = np.concatenate([-5.0 * np.ones(self.du), -5.0 * np.ones(q), (-np.pi / 2.0 + 1e-6) * np.ones(r)])
            xu = np.concatenate([10.0 * np.ones(self.du), 10.0 * np.ones(q), (np.pi / 2.0 - 1e-6) * np.ones(r)])
        elif self._IDX == 2:
            xl = np.concatenate([-5.0 * np.ones(self.du), -5.0 * np.ones(q), 1e-6 * np.ones(r)])
            xu = np.concatenate([10.0 * np.ones(p), np.ones(r), 10.0 * np.ones(q), np.e * np.ones(r)])
        elif self._IDX == 4:
            xl = np.concatenate([-5.0 * np.ones(p), -np.ones(r), -5.0 * np.ones(q), np.zeros(r)])
            xu = np.concatenate([10.0 * np.ones(p), np.ones(r), 10.0 * np.ones(q), np.e * np.ones(r)])
        elif self._IDX in {5, 6, 8}:
            xl = -5.0 * np.ones(d)
            xu = 10.0 * np.ones(d)
        elif self._IDX == 7:
            xl = np.concatenate([-5.0 * np.ones(self.du), -5.0 * np.ones(q), 1e-6 * np.ones(r)])
            xu = np.concatenate([10.0 * np.ones(p), np.ones(r), 10.0 * np.ones(q), np.e * np.ones(r)])
        elif self._IDX == 9:
            xl = np.concatenate([-5.0 * np.ones(self.du), -5.0 * np.ones(q), (-1.0 + 1e-6) * np.ones(r)])
            xu = np.concatenate([10.0 * np.ones(p), np.ones(r), 10.0 * np.ones(q), (np.e - 1.0) * np.ones(r)])
        elif self._IDX == 10:
            xl = np.concatenate([-5.0 * np.ones(self.du), -5.0 * np.ones(q), (-np.pi / 2.0 + 1e-6) * np.ones(r)])
            xu = np.concatenate([10.0 * np.ones(self.du), 10.0 * np.ones(q), (np.pi / 2.0 - 1e-6) * np.ones(r)])
        elif self._IDX == 11:
            xl = np.concatenate([-5.0 * np.ones(p), -np.ones(r), -5.0 * np.ones(q), (1.0 / np.e) * np.ones(r)])
            xu = np.concatenate([10.0 * np.ones(p), np.ones(r), 10.0 * np.ones(q), np.e * np.ones(r)])
        elif self._IDX == 12:
            xl = np.concatenate([-5.0 * np.ones(p), -np.ones(r), -5.0 * np.ones(q), (-np.pi / 4.0 + 1e-6) * np.ones(r)])
            xu = np.concatenate([10.0 * np.ones(p), np.ones(r), 10.0 * np.ones(q), (np.pi / 4.0 - 1e-6) * np.ones(r)])
        else:
            xl = -5.0 * np.ones(d)
            xu = 10.0 * np.ones(d)
        return xl.astype(float), xu.astype(float)

    def _split_common(self, x: np.ndarray):
        p, r, q = self.p, self.r, self.q
        xu1 = x[:, :p]
        xu2 = x[:, p : p + r]
        xl1 = x[:, p + r : p + r + q]
        xl2 = x[:, p + r + q :]
        return xu1, xu2, xl1, xl2

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.clip(_as_2d(x), self.xl, self.xu)
        f = self._calc_f(x)
        out["F"] = f
        g = self._calc_g(x)
        if g is not None and g.shape[1] > 0:
            out["G"] = g

    def _calc_f(self, x: np.ndarray) -> np.ndarray:
        idx = self._IDX
        p, q, r = self.p, self.q, self.r

        if idx == 6:
            xu1 = x[:, :p]
            xu2 = x[:, p : p + r]
            xl1 = x[:, p + r : p + r + q + self.s]
            xl2 = x[:, p + r + q + self.s :]
        else:
            xu1, xu2, xl1, xl2 = self._split_common(x)

        f = np.zeros((x.shape[0], 2), dtype=float)

        if idx == 1:
            t = xu2 - np.tan(xl2)
            f[:, 0] = np.sum(xu1**2, axis=1) + np.sum(xl1**2, axis=1) + np.sum(xu2**2, axis=1) + np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + np.sum(xl1**2, axis=1) + np.sum(t**2, axis=1)

        elif idx == 2:
            t = xu2 - np.log(xl2)
            f[:, 0] = np.sum(xu1**2, axis=1) - np.sum(xl1**2, axis=1) + np.sum(xu2**2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + np.sum(xl1**2, axis=1) + np.sum(t**2, axis=1)

        elif idx == 3:
            t = xu2**2 - np.tan(xl2)
            f[:, 0] = np.sum(xu1**2, axis=1) + np.sum(xl1**2, axis=1) + np.sum(xu2**2, axis=1) + np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + q + np.sum(xl1**2 - np.cos(2.0 * np.pi * xl1), axis=1) + np.sum(t**2, axis=1)

        elif idx == 4:
            t = np.abs(xu2) - np.log(1.0 + xl2)
            f[:, 0] = np.sum(xu1**2, axis=1) - np.sum(xl1**2, axis=1) + np.sum(xu2**2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + q + np.sum(xl1**2 - np.cos(2.0 * np.pi * xl1), axis=1) + np.sum(t**2, axis=1)

        elif idx == 5:
            term2 = _rosenbrock_term(xl1)
            t = np.abs(xu2) - xl2**2
            f[:, 0] = np.sum(xu1**2, axis=1) - term2 + np.sum(xu2**2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + term2 + np.sum(t**2, axis=1)

        elif idx == 6:
            part_a = xl1[:, :q]
            part_b = xl1[:, q:]
            a = part_b[:, 1::2]
            b = part_b[:, 0::2][:, : a.shape[1]]
            term2 = np.sum(part_a**2, axis=1) + np.sum((a - b) ** 2, axis=1)
            t = xu2 - xl2
            f[:, 0] = np.sum(xu1**2, axis=1) - np.sum(part_a**2, axis=1) + np.sum(part_b**2, axis=1) + np.sum(xu2**2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + term2 + np.sum(t**2, axis=1)

        elif idx == 7:
            denom = np.sqrt(np.arange(1, p + 1, dtype=float))[None, :]
            grw = 1.0 + np.sum(xu1**2, axis=1) / 400.0 - np.prod(np.cos(xu1 / np.maximum(denom, 1e-30)), axis=1)
            t = xu2 - np.log(xl2)
            f[:, 0] = grw - np.sum(xl1**2, axis=1) + np.sum(xu2**2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**3, axis=1) + np.sum(xl1**2, axis=1) + np.sum(t**2, axis=1)

        elif idx == 8:
            term2 = _rosenbrock_term(xl1)
            t = xu2 - xl2**3
            ack = 20.0 + np.e - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(xu1**2, axis=1) / max(1, p)))
            ack -= np.exp(np.sum(np.cos(2.0 * np.pi * xu1), axis=1) / max(1, p))
            f[:, 0] = ack - term2 + np.sum(xu2**2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(np.abs(xu1), axis=1) + term2 + np.sum(t**2, axis=1)

        elif idx == 9:
            t = xu2 - np.log(1.0 + xl2)
            f[:, 0] = np.sum(xu1**2, axis=1) - np.sum(xl1**2, axis=1) + np.sum(xu2**2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + np.sum(xl1**2, axis=1) + np.sum(t**2, axis=1)

        elif idx == 10:
            t = xu2 - np.tan(xl2)
            f[:, 0] = np.sum((xu1 - 2.0) ** 2, axis=1) + np.sum(xl1**2, axis=1) + np.sum((xu2 - 2.0) ** 2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + np.sum((xl1 - 2.0) ** 2, axis=1) + np.sum(t**2, axis=1)

        elif idx == 11:
            t = xu2 - np.log(xl2)
            f[:, 0] = np.sum(xu1**2, axis=1) - np.sum(xl1**2, axis=1) + np.sum(xu2**2, axis=1) - np.sum(t**2, axis=1)
            f[:, 1] = np.sum(xu1**2, axis=1) + np.sum(xl1**2, axis=1) + np.sum(t**2, axis=1)

        elif idx == 12:
            t = xu2 - np.tan(xl2)
            f[:, 0] = (
                np.sum((xu1 - 2.0) ** 2, axis=1)
                + np.sum(xl1**2, axis=1)
                + np.sum((xu2 - 2.0) ** 2, axis=1)
                + np.sum(np.tan(np.abs(xl2)), axis=1)
                - np.sum(t**2, axis=1)
            )
            f[:, 1] = np.sum(xu1**2, axis=1) + np.sum((xl1 - 2.0) ** 2, axis=1) + np.sum(t**2, axis=1)

        return f

    def _calc_g(self, x: np.ndarray) -> np.ndarray | None:
        idx = self._IDX
        p, q, r = self.p, self.q, self.r
        xu1, xu2, xl1, xl2 = self._split_common(x)

        if idx <= 8:
            return None

        if idx == 9:
            g1 = -np.sum(xu1**2, axis=1) + np.sum(xu2**2, axis=1) + np.floor(np.sum(xu1**2, axis=1) + np.sum(xu2**2, axis=1) + 0.5)
            g2 = -np.sum(xl1**2, axis=1) + np.sum(xl2**2, axis=1) + np.floor(np.sum(xl1**2, axis=1) + np.sum(xl2**2, axis=1) + 0.5)
            return np.column_stack([g1, g2])

        if idx == 10:
            g = np.zeros((x.shape[0], p + r + q), dtype=float)
            sx1 = np.sum(xu1**3, axis=1)
            sx2 = np.sum(xu2**3, axis=1)
            g[:, :p] = -xu1 - xu1**3 + sx1[:, None] + sx2[:, None]
            g[:, p : p + r] = -xu2 - xu2**3 + sx2[:, None] + sx1[:, None]
            g[:, p + r : p + r + q] = -xl1 - xl1**3 + np.sum(xl1**3, axis=1)[:, None]
            return g

        if idx == 11:
            g1 = -xu2 + 1.0 / np.sqrt(max(1, r)) + np.log(xl2)
            g2 = -np.sum((xu2 - np.log(xl2)) ** 2, axis=1) + 1.0
            return np.column_stack([g1, g2])

        if idx == 12:
            s1 = np.sum(xu1**3, axis=1)
            s2 = np.sum(xu2**3, axis=1)
            g1 = -xu1 - xu1**3 + s1[:, None] + s2[:, None]
            g2 = -xu2 - xu2**3 + s2[:, None] + s1[:, None]
            g3 = -xu2 + np.tan(xl2)
            g4 = -xl1 - xl1**3 + np.sum(xl1**3, axis=1)[:, None]
            g5 = -np.sum((xu2 - np.tan(xl2)) ** 2, axis=1) + 1.0
            return np.column_stack([g1, g2, g3, g4, g5])

        return None


class SMD1(_BaseSMD):
    _IDX = 1


class SMD2(_BaseSMD):
    _IDX = 2


class SMD3(_BaseSMD):
    _IDX = 3


class SMD4(_BaseSMD):
    _IDX = 4


class SMD5(_BaseSMD):
    _IDX = 5


class SMD6(_BaseSMD):
    _IDX = 6

    def __init__(self, n_var: int = 5, max_felower: int = 500, **kwargs):
        d = max(4, int(n_var))
        self.max_felower = int(max_felower)
        self.du = d // 2
        self.dl = d - self.du
        self.r = self.du // 2
        self.p = self.du - self.r
        self.q = int(np.floor((self.dl - self.r) / 2.0 - 1e-6))
        self.s = self.dl - self.r - self.q
        xl = -5.0 * np.ones(d, dtype=float)
        xu = 10.0 * np.ones(d, dtype=float)
        Problem.__init__(self, n_var=d, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu, vtype=float, **kwargs)


class SMD7(_BaseSMD):
    _IDX = 7


class SMD8(_BaseSMD):
    _IDX = 8


class SMD9(_BaseSMD):
    _IDX = 9


class SMD10(_BaseSMD):
    _IDX = 10


class SMD11(_BaseSMD):
    _IDX = 11


class SMD12(_BaseSMD):
    _IDX = 12


_CPU = [f"SMD{i}" for i in range(1, 13)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
