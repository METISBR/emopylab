from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from core.problem import Problem

try:
    import jax.numpy as jnp
except Exception:  # noqa: BLE001
    jnp = None

try:
    import scipy.io as sio
except Exception as _scipy_io_exc:  # noqa: BLE001
    sio = None
    _SCIPY_IO_IMPORT_ERROR = _scipy_io_exc
else:
    _SCIPY_IO_IMPORT_ERROR = None


_CEC2013_REF = (
    "X. Li, K. Tang, M. N. Omidvar, Z. Yang, and K. Qin. Benchmark functions "
    "for the CEC'2013 special session and competition on large-scale global "
    "optimization. RMIT University, Australia, 2013."
)

_S8 = np.array([50, 25, 25, 100, 50, 25, 25, 700], dtype=int)
_S20 = np.array([50, 50, 25, 25, 100, 100, 25, 25, 50, 25, 100, 25, 100, 50, 25, 25, 25, 100, 50, 25], dtype=int)


def _xp(use_jax: bool):
    if use_jax and jnp is not None:
        return jnp
    return np


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _as_2d_np(x: Any) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _community_root() -> Path:
    here = Path(__file__).resolve()
    problems_root = here.parents[1]
    for cand in problems_root.iterdir():
        if cand.is_dir() and (cand / "Single-objective optimization").exists():
            return cand
    raise RuntimeError("Could not locate MATLAB community problems folder.")


def _cec2013_data():
    if sio is None:
        raise RuntimeError(f"scipy.io is required for CEC2013 problems: {_SCIPY_IO_IMPORT_ERROR}")
    path = _community_root() / "Single-objective optimization" / "CEC 2013" / "CEC2013.mat"
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return np.asarray(mat["Data"], dtype=object).reshape(-1)


def _tosz(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x1 = np.zeros_like(x)
    mask = x != 0
    x1[mask] = np.log(np.abs(x[mask]))
    c1 = np.full_like(x, 5.5)
    c1[x > 0] = 10.0
    c2 = np.full_like(x, 3.1)
    c2[x > 0] = 7.9
    return np.sign(x) * np.exp(x1 + 0.049 * (np.sin(c1 * x1) + np.sin(c2 * x1)))


def _tasy(x: np.ndarray, beta: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[1] == 0:
        return x.copy()
    expo = beta * np.linspace(0.0, 1.0, x.shape[1], dtype=float)[None, :]
    z = x.copy()
    pos = x > 0
    if np.any(pos):
        z[pos] = np.power(x[pos], 1.0 + np.broadcast_to(expo * np.sqrt(np.maximum(x, 0.0)), x.shape)[pos])
    return z


def _tdiag(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[1] == 0:
        return x.copy()
    scale = np.power(math.sqrt(float(alpha)), np.linspace(0.0, 1.0, x.shape[1], dtype=float))
    return x * scale[None, :]


def _elliptic(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    w = np.power(10.0, 6.0 * np.linspace(0.0, 1.0, x.shape[1], dtype=float))
    return np.sum(w[None, :] * x**2, axis=1)


def _rastrigin(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)


def _ackley(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    return -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=1))) - np.exp(np.mean(np.cos(2.0 * np.pi * x), axis=1)) + 20.0 + math.e


def _schwefel12(x: np.ndarray) -> np.ndarray:
    return np.sum(np.cumsum(x, axis=1) ** 2, axis=1)


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def _rosenbrock(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1:
        return np.zeros(x.shape[0], dtype=float)
    return np.sum(100.0 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def _rot_for_size(problem, size: int) -> np.ndarray:
    if size == 25:
        return problem.R25
    if size == 50:
        return problem.R50
    if size == 100:
        return problem.R100
    raise ValueError(f"Unsupported block size: {size}")


def _p0(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=int).reshape(-1) - 1


def _block_eval_nonoverlap(
    pop_dec: np.ndarray,
    p: np.ndarray,
    sizes: np.ndarray,
    weights: np.ndarray,
    rot_lookup: Callable[[int], np.ndarray],
    block_func: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    z = np.asarray(pop_dec, dtype=float)
    p_idx = _p0(p)
    f = np.zeros(z.shape[0], dtype=float)
    start = 0
    for i, size in enumerate(np.asarray(sizes, dtype=int).tolist()):
        end = start + size
        loc = p_idx[start:end]
        x = z[:, loc]
        if size in (25, 50, 100):
            x = x @ rot_lookup(size)
        f += float(weights[i]) * block_func(x, i)
        start = end
    return f


class _BaseCEC2013(Problem):
    _USE_JAX = False
    _DATA_INDEX = 1
    _D = 1000
    _LOWER = -100.0
    _UPPER = 100.0

    def __init__(self, **kwargs):
        self._data = _cec2013_data()
        self._item = self._data[int(self._DATA_INDEX) - 1]
        d = int(self._D)
        xl = np.full(d, float(self._LOWER), dtype=float)
        xu = np.full(d, float(self._UPPER), dtype=float)
        super().__init__(n_var=d, n_obj=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _out(self, out, f):
        out["F"] = np.asarray(f, dtype=float).reshape(-1, 1)

    def _prep(self, x) -> np.ndarray:
        return np.clip(_as_2d_np(x), np.asarray(self.xl, dtype=float), np.asarray(self.xu, dtype=float))


class _BaseCEC2013ShiftVector(_BaseCEC2013):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Xopt = np.asarray(self._item, dtype=float).reshape(-1)


class _BaseCEC2013Block(_BaseCEC2013):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        item = self._item
        self.Xopt = np.asarray(item.xopt, dtype=float).reshape(-1)
        self.R25 = np.asarray(item.R25, dtype=float)
        self.R50 = np.asarray(item.R50, dtype=float)
        self.R100 = np.asarray(item.R100, dtype=float)
        self.p = np.asarray(item.p, dtype=int).reshape(-1)


class CEC2013_F1(_BaseCEC2013ShiftVector):
    """Shifted elliptic function. Ref: Li et al. (2013)."""

    _DATA_INDEX = 1
    _LOWER = -100.0
    _UPPER = 100.0

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        f = _elliptic(_tosz(x - self.Xopt[None, :]))
        self._out(out, f)


class CEC2013_F2(_BaseCEC2013ShiftVector):
    """Shifted Rastrigin function. Ref: Li et al. (2013)."""

    _DATA_INDEX = 2
    _LOWER = -5.0
    _UPPER = 5.0

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        z = _tdiag(_tasy(_tosz(x - self.Xopt[None, :]), 0.2), 10.0)
        self._out(out, _rastrigin(z))


class CEC2013_F3(_BaseCEC2013ShiftVector):
    """Shifted Ackley function. Ref: Li et al. (2013)."""

    _DATA_INDEX = 3
    _LOWER = -32.0
    _UPPER = 32.0

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        self._out(out, _ackley(x - self.Xopt[None, :]))


class _BaseCEC2013BlockS8(_BaseCEC2013Block):
    _SIZES = _S8
    _WEIGHTS = np.ones(8, dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        z = x - self.Xopt[None, :]
        f = _block_eval_nonoverlap(
            z,
            self.p,
            self._SIZES,
            np.asarray(self._WEIGHTS, dtype=float),
            lambda size: _rot_for_size(self, size),
            self._block_func,
        )
        self._out(out, f)


class _BaseCEC2013BlockS20(_BaseCEC2013Block):
    _SIZES = _S20
    _WEIGHTS = np.ones(20, dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        z = x - self.Xopt[None, :]
        f = _block_eval_nonoverlap(
            z,
            self.p,
            self._SIZES,
            np.asarray(self._WEIGHTS, dtype=float),
            lambda size: _rot_for_size(self, size),
            self._block_func,
        )
        self._out(out, f)


class CEC2013_F4(_BaseCEC2013BlockS8):
    """7-nonseparable, 1-separable shifted rotated elliptic. Ref: Li et al. (2013)."""

    _DATA_INDEX = 4
    _LOWER = -100.0
    _UPPER = 100.0
    _WEIGHTS = np.array([4.57e1, 1.56e0, 1.85e4, 1.11e-2, 1.36e1, 3.02e-1, 5.96e1, 1.0], dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:
        return _elliptic(_tosz(x))


class CEC2013_F5(_BaseCEC2013BlockS8):
    """7-nonseparable, 1-separable shifted rotated Rastrigin. Ref: Li et al. (2013)."""

    _DATA_INDEX = 5
    _LOWER = -5.0
    _UPPER = 5.0
    _WEIGHTS = np.array([1.81e-1, 9.08e3, 2.43e1, 1.86e-6, 1.77e4, 2.82e-4, 1.53e-2, 1.0], dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:
        return _rastrigin(_tdiag(_tasy(_tosz(x), 0.2), 10.0))


class CEC2013_F6(_BaseCEC2013BlockS8):
    """7-nonseparable, 1-separable shifted rotated Ackley. Ref: Li et al. (2013)."""

    _DATA_INDEX = 6
    _LOWER = -32.0
    _UPPER = 32.0
    _WEIGHTS = np.array([3.53e-2, 5.32e-5, 8.71e-1, 4.95e4, 8.31e-2, 3.48e-5, 2.82e2, 1.0], dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:
        return _ackley(x)


class CEC2013_F7(_BaseCEC2013BlockS8):
    """7-nonseparable, 1-separable shifted rotated Schwefel/Sphere. Ref: Li et al. (2013)."""

    _DATA_INDEX = 7
    _LOWER = -100.0
    _UPPER = 100.0
    _WEIGHTS = np.array([6.80e2, 9.32e-1, 2.12e3, 5.06e-1, 4.35e2, 3.34e4, 2.57e0, 1.0], dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:
        t = _tasy(_tosz(x), 0.2)
        if i < len(self._SIZES) - 1:
            return _schwefel12(t)
        return _sphere(t)


class CEC2013_F8(_BaseCEC2013BlockS20):
    """20-nonseparable shifted rotated elliptic. Ref: Li et al. (2013)."""

    _DATA_INDEX = 8
    _LOWER = -100.0
    _UPPER = 100.0
    _WEIGHTS = np.array([4.63e0, 6.86e-1, 1.14e9, 2.01e0, 7.89e2, 1.63e1, 6.08e0, 6.47e-2, 7.57e-2, 3.57e1, 7.97e-6, 1.08e1, 4.20e-6, 1.92e-3, 1.68e-3, 6.87e2, 1.57e-1, 4.42e-2, 3.54e-1, 6.05e-3], dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:
        return _elliptic(_tosz(x))


class CEC2013_F9(_BaseCEC2013BlockS20):
    """20-nonseparable shifted rotated Rastrigin. Ref: Li et al. (2013)."""

    _DATA_INDEX = 9
    _LOWER = -5.0
    _UPPER = 5.0
    _WEIGHTS = np.array([1.76e3, 5.71e2, 3.36e0, 1.04e0, 6.28e4, 1.73e0, 8.98e-2, 8.07e-4, 1.40e6, 8.72e3, 3.34e-3, 1.35e0, 4.78e-3, 5.09e3, 1.27e1, 3.59e-4, 2.40e-1, 3.96e0, 1.43e-3, 5.23e-3], dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:
        return _rastrigin(_tdiag(_tasy(_tosz(x), 0.2), 10.0))


class CEC2013_F10(_BaseCEC2013BlockS20):
    """20-nonseparable shifted rotated Ackley (per MATLAB source). Ref: Li et al. (2013)."""

    _DATA_INDEX = 10
    _LOWER = -32.0
    _UPPER = 32.0
    _WEIGHTS = np.array([3.13e-1, 1.51e1, 2.32e3, 8.06e-4, 1.14e1, 3.55e0, 3.00e1, 9.98e-1, 1.62e0, 1.51e0, 6.08e-1, 4.46e6, 6.81e-5, 1.36e-1, 7.89e-4, 5.99e4, 1.85e0, 2.48e1, 5.43e-1, 3.92e1], dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:
        return _ackley(x)


class CEC2013_F11(_BaseCEC2013BlockS20):
    """20-nonseparable shifted rotated Schwefel. Ref: Li et al. (2013)."""

    _DATA_INDEX = 11
    _LOWER = -100.0
    _UPPER = 100.0
    _WEIGHTS = np.array([1.61e-2, 1.29e-1, 1.20e-3, 3.49e-1, 3.99e0, 7.45e0, 2.61e0, 1.86e-5, 7.80e-2, 4.95e6, 9.08e2, 1.25e3, 1.28e-4, 2.55e-3, 1.23e-2, 2.25e-1, 1.60e4, 4.15e0, 4.21e3, 8.98e-6], dtype=float)

    def _block_func(self, x: np.ndarray, i: int) -> np.ndarray:
        return _schwefel12(_tasy(_tosz(x), 0.2))


class CEC2013_F12(_BaseCEC2013ShiftVector):
    """Shifted Rosenbrock function. Ref: Li et al. (2013)."""

    _DATA_INDEX = 12
    _LOWER = -100.0
    _UPPER = 100.0

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        self._out(out, _rosenbrock(x - self.Xopt[None, :]))


class CEC2013_F13(_BaseCEC2013Block):
    """Shifted Schwefel with conforming overlapping subcomponents. Ref: Li et al. (2013)."""

    _DATA_INDEX = 13
    _D = 905
    _LOWER = -100.0
    _UPPER = 100.0
    _WEIGHTS = np.array([4.35e-1, 9.92e-3, 5.43e-2, 2.94e1, 1.15e4, 2.41e1, 3.45e0, 2.33e0, 1.77e-3, 2.54e-2, 2.00e1, 3.67e-4, 1.36e-3, 3.87e-2, 8.89e1, 5.79e4, 8.49e-3, 7.36e-2, 6.88e-1, 1.19e5], dtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        z = x - self.Xopt[None, :]
        p_idx = _p0(self.p)
        f = np.zeros(x.shape[0], dtype=float)
        csum = np.cumsum(_S20)
        prev = np.concatenate([[0], csum[:-1]])
        for i, size in enumerate(_S20.tolist()):
            start = prev[i] - i * 5
            end = csum[i] - i * 5
            loc = p_idx[start:end]
            blk = z[:, loc] @ _rot_for_size(self, int(size))
            f += float(self._WEIGHTS[i]) * _schwefel12(_tasy(_tosz(blk), 0.2))
        self._out(out, f)


class CEC2013_F14(_BaseCEC2013Block):
    """Shifted Schwefel with conflicting overlapping subcomponents. Ref: Li et al. (2013)."""

    _DATA_INDEX = 14
    _D = 905
    _LOWER = -100.0
    _UPPER = 100.0
    _WEIGHTS = np.array([4.75e-1, 4.99e5, 3.28e2, 3.23e-1, 1.36e2, 9.03e0, 9.24e-2, 1.10e-4, 9.37e-3, 3.00e2, 4.94e0, 8.14e1, 6.54e-1, 1.16e1, 2.86e6, 8.58e-5, 2.36e1, 4.81e-2, 1.43e0, 1.22e1], dtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        p_idx = _p0(self.p)
        f = np.zeros(x.shape[0], dtype=float)
        csum = np.cumsum(_S20)
        prev = np.concatenate([[0], csum[:-1]])
        for i, size in enumerate(_S20.tolist()):
            loc1_start = prev[i] - i * 5
            loc1_end = csum[i] - i * 5
            loc1 = p_idx[loc1_start:loc1_end]
            loc2 = np.arange(prev[i], csum[i], dtype=int)
            blk = (x[:, loc1] - self.Xopt[loc2][None, :]) @ _rot_for_size(self, int(size))
            f += float(self._WEIGHTS[i]) * _schwefel12(_tasy(_tosz(blk), 0.2))
        self._out(out, f)


class CEC2013_F15(_BaseCEC2013ShiftVector):
    """Shifted Schwefel function. Ref: Li et al. (2013)."""

    _DATA_INDEX = 15
    _LOWER = -100.0
    _UPPER = 100.0

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prep(x)
        self._out(out, _schwefel12(_tasy(_tosz(x - self.Xopt[None, :]), 0.2)))


_CPU = [f"CEC2013_F{i}" for i in range(1, 16)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU + [f"{name}_JAX" for name in _CPU]

