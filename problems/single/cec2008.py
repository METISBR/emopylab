from __future__ import annotations

import math
from pathlib import Path
from typing import Any

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


_CEC2008_REF = (
    "K. Tang, X. Yao, P. N. Suganthan, C. MacNish, Y.-P. Chen, C.-M. Chen, "
    "and Z. Yang. Benchmark functions for the CEC'2008 special session and "
    "competition on large scale global optimization. Nature Inspired "
    "Computation and Applications Laboratory, USTC, China, 2007."
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


def _problems_matlab_community_root() -> Path:
    here = Path(__file__).resolve()
    problems_root = here.parents[1]
    for cand in problems_root.iterdir():
        if cand.is_dir() and (cand / "Single-objective optimization").exists():
            return cand
    raise RuntimeError("Could not locate MATLAB community problems folder.")


def _cec2008_data() -> np.ndarray:
    if sio is None:
        raise RuntimeError(f"scipy.io is required for CEC2008 problems: {_SCIPY_IO_IMPORT_ERROR}")
    path = _problems_matlab_community_root() / "Single-objective optimization" / "CEC 2008" / "CEC2008.mat"
    data = sio.loadmat(str(path), squeeze_me=True)["Data"]
    return np.asarray(data, dtype=object).reshape(-1)


class _BaseCEC2008(Problem):
    _USE_JAX = False
    _DATA_IDX = 0
    _LOWER = -100.0
    _UPPER = 100.0

    def __init__(self, n_var: int = 100, **kwargs):
        self._data = _cec2008_data()
        self.O = np.asarray(self._data[self._DATA_IDX], dtype=float).reshape(-1)
        d = min(int(max(1, n_var)), int(self.O.shape[0]))
        super().__init__(
            n_var=d,
            n_obj=1,
            xl=np.full(d, float(self._LOWER), dtype=float),
            xu=np.full(d, float(self._UPPER), dtype=float),
            vtype=float,
            **kwargs,
        )

    def _xp(self):
        return _xp(self._USE_JAX)

    def _prepare_shifted(self, x):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        o = xp.asarray(self.O[: x.shape[1]], dtype=float)
        z = x - o[xp.newaxis, :]
        return xp, x, z

    def _out(self, out, f):
        out["F"] = _to_numpy(f).reshape(-1, 1)


class CEC2008_F1(_BaseCEC2008):
    """Shifted sphere function. Ref: CEC 2008 benchmark definitions."""

    _DATA_IDX = 0
    _LOWER = -100.0
    _UPPER = 100.0

    def _evaluate(self, x, out, *args, **kwargs):
        xp, _x, z = self._prepare_shifted(x)
        self._out(out, xp.sum(z**2, axis=1))


class CEC2008_F2(_BaseCEC2008):
    """Shifted Schwefel's function (max abs). Ref: CEC 2008 benchmark definitions."""

    _DATA_IDX = 1
    _LOWER = -100.0
    _UPPER = 100.0

    def _evaluate(self, x, out, *args, **kwargs):
        xp, _x, z = self._prepare_shifted(x)
        self._out(out, xp.max(xp.abs(z), axis=1))


class CEC2008_F3(_BaseCEC2008):
    """Shifted Rosenbrock's function. Ref: CEC 2008 benchmark definitions."""

    _DATA_IDX = 2
    _LOWER = -100.0
    _UPPER = 100.0

    def _evaluate(self, x, out, *args, **kwargs):
        xp, _x, z = self._prepare_shifted(x)
        f = xp.sum(100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1.0) ** 2, axis=1)
        self._out(out, f)


class CEC2008_F4(_BaseCEC2008):
    """Shifted Rastrigin's function. Ref: CEC 2008 benchmark definitions."""

    _DATA_IDX = 3
    _LOWER = -5.0
    _UPPER = 5.0

    def _evaluate(self, x, out, *args, **kwargs):
        xp, _x, z = self._prepare_shifted(x)
        self._out(out, xp.sum(z**2 - 10.0 * xp.cos(2.0 * xp.pi * z) + 10.0, axis=1))


class CEC2008_F5(_BaseCEC2008):
    """Shifted Griewank's function. Ref: CEC 2008 benchmark definitions."""

    _DATA_IDX = 4
    _LOWER = -600.0
    _UPPER = 600.0

    def _evaluate(self, x, out, *args, **kwargs):
        xp, _x, z = self._prepare_shifted(x)
        idx = xp.sqrt(xp.arange(1, z.shape[1] + 1, dtype=float))
        f = xp.sum(z**2, axis=1) / 4000.0 - xp.prod(xp.cos(z / idx[xp.newaxis, :]), axis=1) + 1.0
        self._out(out, f)


class CEC2008_F6(_BaseCEC2008):
    """Shifted Ackley's function. Ref: CEC 2008 benchmark definitions."""

    _DATA_IDX = 5
    _LOWER = -32.0
    _UPPER = 32.0

    def _evaluate(self, x, out, *args, **kwargs):
        xp, _x, z = self._prepare_shifted(x)
        f = -20.0 * xp.exp(-0.2 * xp.sqrt(xp.mean(z**2, axis=1))) - xp.exp(xp.mean(xp.cos(2.0 * xp.pi * z), axis=1)) + 20.0 + math.e
        self._out(out, f)


class CEC2008_F7(_BaseCEC2008):
    """FastFractal 'DoubleDip' function. Ref: CEC 2008 benchmark definitions."""

    _DATA_IDX = 6
    _LOWER = -1.0
    _UPPER = 1.0

    def __init__(self, n_var: int = 100, **kwargs):
        self._data = _cec2008_data()
        ran = np.asarray(self._data[self._DATA_IDX], dtype=float)
        self.ran11 = ran[0, :].reshape(-1)
        self.ran12 = ran[1, :].reshape(-1)
        self.ran2 = ran[2, :].reshape(-1)
        d = min(int(max(1, n_var)), 1000)
        Problem.__init__(
            self,
            n_var=d,
            n_obj=1,
            xl=np.full(d, self._LOWER, dtype=float),
            xu=np.full(d, self._UPPER, dtype=float),
            vtype=float,
            **kwargs,
        )

    def _xp(self):
        return _xp(self._USE_JAX)

    def _evaluate(self, x, out, *args, **kwargs):
        x_np = _to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        x_np = np.clip(x_np, self.xl, self.xu)
        f = np.zeros(x_np.shape[0], dtype=float)
        for i in range(x_np.shape[0]):
            loc1 = 0
            loc2 = 0
            total = 0.0
            for j in range(x_np.shape[1]):
                xij = float(x_np[i, j])
                for k in range(1, 4):
                    for _k1 in range(2 ** (k - 1)):
                        loc2 += 1
                        reps = int(round(float(self.ran2[loc2 - 1])))
                        for _k2 in range(reps):
                            loc1 += 1
                            if abs(xij) < 0.5:
                                d = xij - float(self.ran11[loc1 - 1])
                                numer = -6144.0 * d**6 + 3088.0 * d**4 - 392.0 * d**2 + 1.0
                                total += numer / (2 ** (k - 1)) / (2.0 - float(self.ran12[loc1 - 1]))
            f[i] = total
        out["F"] = f.reshape(-1, 1)


_CPU = [f"CEC2008_F{i}" for i in range(1, 8)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU + [f"{name}_JAX" for name in _CPU]
