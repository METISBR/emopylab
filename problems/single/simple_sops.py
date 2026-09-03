from __future__ import annotations

import math
from typing import Any

import numpy as np
from core.problem import Problem

try:
    import jax.numpy as jnp
except Exception:  # noqa: BLE001
    jnp = None


_YAO_1999_REF = (
    "X. Yao, Y. Liu, and G. Lin. Evolutionary programming made faster. "
    "IEEE Transactions on Evolutionary Computation, 1999, 3(2): 82-102."
)


def _xp(use_jax: bool):
    if use_jax and jnp is not None:
        return jnp
    return np


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _as_2d(x: Any, xp):
    arr = xp.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[xp.newaxis, :]
    return arr


def _clip_x(x: Any, xl: Any, xu: Any, xp):
    arr = _as_2d(x, xp)
    return xp.clip(arr, xp.asarray(xl, dtype=float), xp.asarray(xu, dtype=float))


def _u_penalty(x: np.ndarray | Any, a: float, k: float, m: float, xp) -> Any:
    ax = xp.abs(x)
    return xp.where(ax > a, k * (ax - a) ** m, 0.0)


def _ones_row(value: float, d: int) -> np.ndarray:
    return np.full(int(d), float(value), dtype=float)


class _BaseSimpleSOP(Problem):
    _USE_JAX = False

    def _xp(self):
        return _xp(self._USE_JAX)

    def _prepare(self, x):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        return xp, x

    def _out(self, out: dict[str, Any], f):
        out["F"] = _to_numpy(f).reshape(-1, 1)


class SOP_F1(_BaseSimpleSOP):
    """Sphere function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-100, n_var), xu=_ones_row(100, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        self._out(out, xp.sum(x**2, axis=1))


class SOP_F2(_BaseSimpleSOP):
    """Schwefel's function 2.22. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-10, n_var), xu=_ones_row(10, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        self._out(out, xp.sum(xp.abs(x), axis=1) + xp.prod(xp.abs(x), axis=1))


class SOP_F3(_BaseSimpleSOP):
    """Schwefel's function 1.2. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-100, n_var), xu=_ones_row(100, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        self._out(out, xp.sum(xp.cumsum(x, axis=1) ** 2, axis=1))


class SOP_F4(_BaseSimpleSOP):
    """Schwefel's function 2.21. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-100, n_var), xu=_ones_row(100, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        self._out(out, xp.max(xp.abs(x), axis=1))


class SOP_F5(_BaseSimpleSOP):
    """Generalized Rosenbrock's function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-30, n_var), xu=_ones_row(30, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        f = xp.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)
        self._out(out, f)


class SOP_F6(_BaseSimpleSOP):
    """Step function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-100, n_var), xu=_ones_row(100, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        self._out(out, xp.sum(xp.floor(x + 0.5) ** 2, axis=1))


class SOP_F7(_BaseSimpleSOP):
    """Quartic function with noise. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-1.28, n_var), xu=_ones_row(1.28, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        idx = xp.arange(1, self.n_var + 1, dtype=float)
        base = xp.sum(idx[xp.newaxis, :] * x**4, axis=1)
        noise = np.random.random(size=(int(x.shape[0]),))
        self._out(out, base + xp.asarray(noise, dtype=float))


class SOP_F8(_BaseSimpleSOP):
    """Generalized Schwefel's function 2.26. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-500, n_var), xu=_ones_row(500, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        self._out(out, -xp.sum(x * xp.sin(xp.sqrt(xp.abs(x))), axis=1))


class SOP_F9(_BaseSimpleSOP):
    """Generalized Rastrigin's function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-5.12, n_var), xu=_ones_row(5.12, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        self._out(out, xp.sum(x**2 - 10.0 * xp.cos(2.0 * xp.pi * x) + 10.0, axis=1))


class SOP_F10(_BaseSimpleSOP):
    """Ackley's function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-32, n_var), xu=_ones_row(32, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        term1 = -20.0 * xp.exp(-0.2 * xp.sqrt(xp.mean(x**2, axis=1)))
        term2 = -xp.exp(xp.mean(xp.cos(2.0 * xp.pi * x), axis=1))
        self._out(out, term1 + term2 + 20.0 + math.e)


class SOP_F11(_BaseSimpleSOP):
    """Generalized Griewank's function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-600, n_var), xu=_ones_row(600, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        idx = xp.sqrt(xp.arange(1, self.n_var + 1, dtype=float))
        prod_term = xp.prod(xp.cos(x / idx[xp.newaxis, :]), axis=1)
        self._out(out, xp.sum(x**2, axis=1) / 4000.0 - prod_term + 1.0)


class SOP_F12(_BaseSimpleSOP):
    """Generalized penalized function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-50, n_var), xu=_ones_row(50, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        inner = 10.0 * xp.sin(xp.pi * x[:, 0]) ** 2
        inner += xp.sum((x[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * xp.sin(xp.pi * x[:, 1:]) ** 2), axis=1)
        inner += (x[:, -1] - 1.0) ** 2
        penalty = xp.sum(_u_penalty(x, 10.0, 100.0, 4.0, xp), axis=1)
        self._out(out, (xp.pi / 30.0) * inner + penalty)


class SOP_F13(_BaseSimpleSOP):
    """Generalized penalized function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=int(n_var), n_obj=1, xl=_ones_row(-50, n_var), xu=_ones_row(50, n_var), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        inner = xp.sin(3.0 * xp.pi * x[:, 0]) ** 2
        inner += xp.sum((x[:, :-1] - 1.0) ** 2 * (1.0 + xp.sin(3.0 * xp.pi * x[:, 1:]) ** 2), axis=1)
        inner += (x[:, -1] - 1.0) ** 2 * (1.0 + xp.sin(2.0 * xp.pi * x[:, -1]) ** 2)
        penalty = xp.sum(_u_penalty(x, 5.0, 100.0, 4.0, xp), axis=1)
        self._out(out, 0.1 * inner + penalty)


class SOP_F14(_BaseSimpleSOP):
    """Shekel's foxholes function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 2, **kwargs):
        super().__init__(n_var=2, n_obj=1, xl=_ones_row(-65.536, 2), xu=_ones_row(65.536, 2), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        a = np.tile(np.arange(-32.0, 33.0, 16.0), (5, 1))
        a = np.vstack([a.T.reshape(1, -1), a.reshape(1, -1)])
        a = xp.asarray(a, dtype=float)  # shape (2,25)
        idx = xp.arange(1.0, 26.0, dtype=float)
        diff = x[:, :, xp.newaxis] - a[xp.newaxis, :, :]
        denom = idx[xp.newaxis, :] + xp.sum(diff**6, axis=1)
        f = 1.0 / (1.0 / 500.0 + xp.sum(1.0 / denom, axis=1))
        self._out(out, f)


class SOP_F15(_BaseSimpleSOP):
    """Kowalik's function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 4, **kwargs):
        super().__init__(n_var=4, n_obj=1, xl=_ones_row(-5.0, 4), xu=_ones_row(5.0, 4), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        a = xp.asarray([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246], dtype=float)
        b = 1.0 / xp.asarray([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
        x1, x2, x3, x4 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        num = x1 * (b[xp.newaxis, :] ** 2 + b[xp.newaxis, :] * x2)
        den = b[xp.newaxis, :] ** 2 + b[xp.newaxis, :] * x3 + x4
        f = xp.sum((a[xp.newaxis, :] - num / den) ** 2, axis=1)
        self._out(out, f)


class SOP_F16(_BaseSimpleSOP):
    """Six-hump camel-back function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 2, **kwargs):
        super().__init__(n_var=2, n_obj=1, xl=_ones_row(-5.0, 2), xu=_ones_row(5.0, 2), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        x1, x2 = x[:, 0], x[:, 1]
        f = 4.0 * x1**2 - 2.1 * x1**4 + x1**6 / 3.0 + x1 * x2 - 4.0 * x2**2 + 4.0 * x2**4
        self._out(out, f)


class SOP_F17(_BaseSimpleSOP):
    """Branin function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 2, **kwargs):
        super().__init__(n_var=2, n_obj=1, xl=np.asarray([-5.0, 0.0]), xu=np.asarray([10.0, 15.0]), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        x1, x2 = x[:, 0], x[:, 1]
        f = (x2 - 5.1 / (4.0 * xp.pi**2) * x1**2 + 5.0 / xp.pi * x1 - 6.0) ** 2
        f += 10.0 * (1.0 - 1.0 / (8.0 * xp.pi)) * xp.cos(x1) + 10.0
        self._out(out, f)


class SOP_F18(_BaseSimpleSOP):
    """Goldstein-Price function. Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 2, **kwargs):
        super().__init__(n_var=2, n_obj=1, xl=np.asarray([-2.0, -2.0]), xu=np.asarray([2.0, 2.0]), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        x1, x2 = x[:, 0], x[:, 1]
        t1 = 1.0 + (x1 + x2 + 1.0) ** 2 * (19.0 - 14.0 * x1 + 3.0 * x1**2 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2**2)
        t2 = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (18.0 - 32.0 * x1 + 12.0 * x1**2 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2**2)
        self._out(out, t1 * t2)


class SOP_F19(_BaseSimpleSOP):
    """Hartman's family (3D). Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 3, **kwargs):
        super().__init__(n_var=3, n_obj=1, xl=_ones_row(0.0, 3), xu=_ones_row(1.0, 3), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        a = xp.asarray([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]], dtype=float)
        c = xp.asarray([1.0, 1.2, 3.0, 3.2], dtype=float)
        p = xp.asarray(
            [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]],
            dtype=float,
        )
        diff = x[:, xp.newaxis, :] - p[xp.newaxis, :, :]
        f = -xp.sum(c[xp.newaxis, :] * xp.exp(-xp.sum(a[xp.newaxis, :, :] * diff**2, axis=2)), axis=1)
        self._out(out, f)


class SOP_F20(_BaseSimpleSOP):
    """Hartman's family (6D). Ref: Yao et al. (1999)."""

    def __init__(self, n_var: int = 6, **kwargs):
        super().__init__(n_var=6, n_obj=1, xl=_ones_row(0.0, 6), xu=_ones_row(1.0, 6), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        a = xp.asarray(
            [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]],
            dtype=float,
        )
        c = xp.asarray([1.0, 1.2, 3.0, 3.2], dtype=float)
        p = xp.asarray(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ],
            dtype=float,
        )
        diff = x[:, xp.newaxis, :] - p[xp.newaxis, :, :]
        f = -xp.sum(c[xp.newaxis, :] * xp.exp(-xp.sum(a[xp.newaxis, :, :] * diff**2, axis=2)), axis=1)
        self._out(out, f)


class _ShekelFamilyBase(_BaseSimpleSOP):
    _A = None
    _C = None

    def __init__(self, n_var: int = 4, **kwargs):
        super().__init__(n_var=4, n_obj=1, xl=_ones_row(0.0, 4), xu=_ones_row(10.0, 4), vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        a = xp.asarray(self._A, dtype=float)
        c = xp.asarray(self._C, dtype=float).reshape(-1)
        diff = x[:, xp.newaxis, :] - a[xp.newaxis, :, :]
        f = -xp.sum(1.0 / (xp.sum(diff**2, axis=2) + c[xp.newaxis, :]), axis=1)
        self._out(out, f)


class SOP_F21(_ShekelFamilyBase):
    """Shekel's family (m=5). Ref: Yao et al. (1999)."""

    _A = np.asarray([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7]], dtype=float)
    _C = np.asarray([0.1, 0.2, 0.2, 0.4, 0.4], dtype=float)


class SOP_F22(_ShekelFamilyBase):
    """Shekel's family (m=7). Ref: Yao et al. (1999)."""

    _A = np.asarray(
        [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3]],
        dtype=float,
    )
    _C = np.asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3], dtype=float)


class SOP_F23(_ShekelFamilyBase):
    """Shekel's family (m=10). Ref: Yao et al. (1999)."""

    _A = np.asarray(
        [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ],
        dtype=float,
    )
    _C = np.asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5], dtype=float)


# Note: The original MATLAB community Simple SOPs family in this folder indexes 1..23 and
# includes several classic fixed-dimension test functions. SOP_F14..SOP_F23 are
# fixed-dimension; SOP_F1..SOP_F13 are scalable unless specified otherwise.
_CPU_CLASSES = [
    "SOP_F1",
    "SOP_F2",
    "SOP_F3",
    "SOP_F4",
    "SOP_F5",
    "SOP_F6",
    "SOP_F7",
    "SOP_F8",
    "SOP_F9",
    "SOP_F10",
    "SOP_F11",
    "SOP_F12",
    "SOP_F13",
    "SOP_F14",
    "SOP_F15",
    "SOP_F16",
    "SOP_F17",
    "SOP_F18",
    "SOP_F19",
    "SOP_F20",
    "SOP_F21",
    "SOP_F22",
    "SOP_F23",
]

for _name in _CPU_CLASSES:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU_CLASSES + [f"{name}_JAX" for name in _CPU_CLASSES]
