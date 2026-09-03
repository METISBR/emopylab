from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from core.problem import Problem

try:
    import jax.numpy as jnp
except Exception:  # noqa: BLE001
    jnp = None


_MFEA2_REF = (
    "K. K. Bali, Y. Ong, A. Gupta, and P. S. Tan. Multifactorial evolutionary "
    "algorithm with online transfer parameter estimation: MFEA-II. IEEE Transactions "
    "on Evolutionary Computation, 2020, 24(1): 69-83."
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


def _parse_subd(subd: Iterable[int] | None, n_var: int | None) -> tuple[int, int]:
    if subd is not None:
        vals = [int(v) for v in subd]
        if len(vals) != 2:
            raise ValueError("subd must have exactly two task dimensions.")
        return max(1, vals[0]), max(1, vals[1])
    if n_var is None:
        return 50, 50
    d = max(2, int(n_var))
    k = max(1, d - 1)
    return k, k


def _griewank(x, xp):
    idx = xp.sqrt(xp.arange(1, x.shape[1] + 1, dtype=float))
    return xp.sum(x**2, axis=1) / 4000.0 - xp.prod(xp.cos(x / idx[xp.newaxis, :]), axis=1) + 1.0


def _rastrigin(x, xp):
    return xp.sum(x**2 - 10.0 * xp.cos(2.0 * xp.pi * x) + 10.0, axis=1)


def _ackley(x, xp):
    return -20.0 * xp.exp(-0.2 * xp.sqrt(xp.mean(x**2, axis=1))) - xp.exp(xp.mean(xp.cos(2.0 * xp.pi * x), axis=1)) + 20.0 + math.e


def _schwefel226(x, xp):
    return -xp.sum(x * xp.sin(xp.sqrt(xp.abs(x))), axis=1)


def _rosenbrock(x, xp):
    return xp.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def _weierstrass(x, xp):
    # a=0.5, b=3, kmax=20 as in the MATLAB source.
    a = 0.5
    b = 3.0
    k = xp.arange(0, 21, dtype=float)
    ak = a**k
    bk = b**k
    term1 = xp.sum(
        xp.sum(ak[xp.newaxis, xp.newaxis, :] * xp.cos(2.0 * xp.pi * bk[xp.newaxis, xp.newaxis, :] * (x[:, :, xp.newaxis] + 0.5)), axis=2),
        axis=1,
    )
    term2 = x.shape[1] * xp.sum(ak * xp.cos(2.0 * xp.pi * bk * 0.5))
    return term1 - term2


class _BaseMultitaskSOP(Problem):
    _USE_JAX = False
    _TASK1_BOUNDS = (-1.0, 1.0)
    _TASK2_BOUNDS = (-1.0, 1.0)

    def __init__(self, n_var: int | None = None, subd: tuple[int, int] | list[int] | None = None, **kwargs):
        subd1, subd2 = _parse_subd(subd, n_var)
        self.SubD = (subd1, subd2)
        d = max(subd1, subd2) + 1
        xl = np.concatenate([np.zeros(d - 1, dtype=float), np.asarray([1.0])])
        xu = np.concatenate([np.ones(d - 1, dtype=float), np.asarray([2.0])])
        super().__init__(n_var=int(d), n_obj=1, xl=xl, xu=xu, vtype=float, **kwargs)

        self.L1 = np.full(subd1, float(self._TASK1_BOUNDS[0]), dtype=float)
        self.U1 = np.full(subd1, float(self._TASK1_BOUNDS[1]), dtype=float)
        self.L2 = np.full(subd2, float(self._TASK2_BOUNDS[0]), dtype=float)
        self.U2 = np.full(subd2, float(self._TASK2_BOUNDS[1]), dtype=float)

    def _xp(self):
        return _xp(self._USE_JAX)

    def _prepare(self, x):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        return xp, x

    def _decode_task(self, x_row: np.ndarray, task: int) -> np.ndarray:
        if task == 1:
            d = self.SubD[0]
            return self.L1 + x_row[:d] * (self.U1 - self.L1)
        d = self.SubD[1]
        return self.L2 + x_row[:d] * (self.U2 - self.L2)

    def _task1(self, x_task, xp):  # pragma: no cover - overridden
        raise NotImplementedError

    def _task2(self, x_task, xp):  # pragma: no cover - overridden
        raise NotImplementedError

    def _evaluate(self, x, out, *args, **kwargs):
        xp, x = self._prepare(x)
        # Task selector is encoded in the last variable with bounds [1,2].
        task_selector = np.rint(_to_numpy(x[:, -1])).astype(int)
        task_selector = np.clip(task_selector, 1, 2)

        f = np.zeros(int(x.shape[0]), dtype=float)
        x_np = _to_numpy(x)
        for i in range(x_np.shape[0]):
            task = int(task_selector[i])
            if task == 1:
                x_task = self._decode_task(x_np[i, :-1], 1)
                val = self._task1(_as_2d(x_task, xp), xp)
            else:
                x_task = self._decode_task(x_np[i, :-1], 2)
                val = self._task2(_as_2d(x_task, xp), xp)
            f[i] = float(_to_numpy(val).reshape(-1)[0])

        out["F"] = f.reshape(-1, 1)


class CI_HS(_BaseMultitaskSOP):
    """Multitasking problem (Griewank + Rastrigin). Ref: Bali et al. (2020)."""

    _TASK1_BOUNDS = (-600.0, 600.0)
    _TASK2_BOUNDS = (-5.12, 5.12)

    def _task1(self, x_task, xp):
        return _griewank(x_task, xp)

    def _task2(self, x_task, xp):
        return _rastrigin(x_task, xp)


class CI_LS(_BaseMultitaskSOP):
    """Multitasking problem (Ackley + Schwefel). Ref: Bali et al. (2020)."""

    _TASK1_BOUNDS = (-32.0, 32.0)
    _TASK2_BOUNDS = (-500.0, 500.0)

    def _task1(self, x_task, xp):
        return _ackley(x_task, xp)

    def _task2(self, x_task, xp):
        return _schwefel226(x_task, xp)


class CI_MS(_BaseMultitaskSOP):
    """Multitasking problem (Ackley + Rastrigin). Ref: Bali et al. (2020)."""

    _TASK1_BOUNDS = (-32.0, 32.0)
    _TASK2_BOUNDS = (-5.12, 5.12)

    def _task1(self, x_task, xp):
        return _ackley(x_task, xp)

    def _task2(self, x_task, xp):
        return _rastrigin(x_task, xp)


class NI_HS(_BaseMultitaskSOP):
    """Multitasking problem (Rosenbrock + Rastrigin). Ref: Bali et al. (2020)."""

    _TASK1_BOUNDS = (-30.0, 30.0)
    _TASK2_BOUNDS = (-5.12, 5.12)

    def _task1(self, x_task, xp):
        return _rosenbrock(x_task, xp)

    def _task2(self, x_task, xp):
        return _rastrigin(x_task, xp)


class NI_MS(_BaseMultitaskSOP):
    """Multitasking problem (Griewank + Weierstrass). Ref: Bali et al. (2020)."""

    _TASK1_BOUNDS = (-600.0, 600.0)
    _TASK2_BOUNDS = (-0.5, 0.5)

    def _task1(self, x_task, xp):
        return _griewank(x_task, xp)

    def _task2(self, x_task, xp):
        return _weierstrass(x_task, xp)


_CPU = ["CI_HS", "CI_LS", "CI_MS", "NI_HS", "NI_MS"]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU + [f"{name}_JAX" for name in _CPU]

