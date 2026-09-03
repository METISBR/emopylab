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


_CEC2020_REF = (
    "C. T. Yue, K. V. Price, P. N. Suganthan, J. J. Liang, M. Z. Ali, "
    "B. Y. Qu, N. H. Awad, and P. P. Biswas. Problem definitions and "
    "evaluation criteria for the CEC 2020 special session and competition "
    "on single objective bound constrained numerical optimization. "
    "Zhengzhou University and Nanyang Technological University, 2019."
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


def _problems_matlab_community_root() -> Path:
    here = Path(__file__).resolve()
    problems_root = here.parents[1]
    for cand in problems_root.iterdir():
        if cand.is_dir() and (cand / "Single-objective optimization").exists():
            return cand
    raise RuntimeError("Could not locate MATLAB community problems folder.")


def _cec2020_data():
    if sio is None:
        raise RuntimeError(f"scipy.io is required for CEC2020 problems: {_SCIPY_IO_IMPORT_ERROR}")
    path = _problems_matlab_community_root() / "Single-objective optimization" / "CEC 2020" / "CEC2020.mat"
    data = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return np.asarray(data["Data"], dtype=object).reshape(-1)


def _select_dim(n_var: int | None) -> int:
    if n_var is None or int(n_var) < 10:
        return 5
    if int(n_var) < 15:
        return 10
    if int(n_var) < 20:
        return 15
    return 20


def _get_field_dim(item, prefix: str, d: int):
    return np.asarray(getattr(item, f"{prefix}_{int(d)}"), dtype=float)


def _clip_bounds_np(x: Any, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    x_np = _to_numpy(x)
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    return np.clip(x_np, xl, xu)


def _elliptic(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    expo = np.arange(0, x.shape[1], dtype=float) / (x.shape[1] - 1 + 1e-6)
    return np.sum((1e6**expo)[None, :] * x**2, axis=1)


def _rastrigin(x: np.ndarray) -> np.ndarray:
    xx = 0.0512 * x
    return np.sum(xx**2 - 10.0 * np.cos(2.0 * np.pi * xx) + 10.0, axis=1)


def _griewank(x: np.ndarray) -> np.ndarray:
    xx = 6.0 * x
    if xx.shape[1] == 0:
        return np.zeros(xx.shape[0], dtype=float)
    idx = np.sqrt(np.arange(1, xx.shape[1] + 1, dtype=float))
    return np.sum(xx**2, axis=1) / 4000.0 - np.prod(np.cos(xx) / idx[None, :], axis=1) + 1.0


def _ackley(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    return -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=1))) - np.exp(np.mean(np.cos(2.0 * np.pi * x), axis=1)) + 20.0 + math.e


def _discus(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    rest = np.sum(x[:, 1:] ** 2, axis=1) if x.shape[1] > 1 else 0.0
    return 1e6 * x[:, 0] ** 2 + rest


def _rosenbrock_scaled(x: np.ndarray) -> np.ndarray:
    xx = 0.02048 * x + 1.0
    if xx.shape[1] <= 1:
        return np.zeros(xx.shape[0], dtype=float)
    return np.sum(100.0 * (xx[:, :-1] ** 2 - xx[:, 1:]) ** 2 + (xx[:, :-1] - 1.0) ** 2, axis=1)


def _schwefel_shifted(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    d = float(x.shape[1])
    z = 10.0 * x + 4.2097e2
    g = z * np.sin(np.sqrt(np.abs(z)))
    mask_hi = z > 500.0
    if np.any(mask_hi):
        temp = 500.0 - np.mod(z[mask_hi], 500.0)
        g[mask_hi] = temp * np.sin(np.sqrt(np.abs(temp))) - (z[mask_hi] - 500.0) ** 2 / 10000.0 / d
    mask_lo = z < -500.0
    if np.any(mask_lo):
        temp = np.mod(np.abs(z[mask_lo]), 500.0) - 500.0
        # Matches the MATLAB source (linear term, not squared).
        g[mask_lo] = temp * np.sin(np.sqrt(np.abs(temp))) - (z[mask_lo] - 500.0) / 10000.0 / d
    return 418.9829 * d - np.sum(g, axis=1)


def _schaffer(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    xx = x**2
    nxt = xx[:, [*range(1, xx.shape[1]), 0]]
    s = xx + nxt
    return np.sum(0.5 + (np.sin(np.sqrt(s)) ** 2 - 0.5) / (1.0 + 0.001 * s) ** 2, axis=1)


def _hgbat(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    xx = 0.05 * x - 1.0
    s2 = np.sum(xx**2, axis=1)
    s1 = np.sum(xx, axis=1)
    d = float(x.shape[1])
    return np.sqrt(np.abs(s2**2 - s1**2)) + (0.5 * s2 + s1) / d + 0.5


def _happycat(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    xx = 0.05 * x
    s2 = np.sum(xx**2, axis=1)
    s1 = np.sum(xx, axis=1)
    d = float(x.shape[1])
    return np.abs(s2 - d) ** 0.25 + (0.5 * s2 + s1) / d + 0.5


def _partition_indices(s_perm_1based: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    s = np.asarray(s_perm_1based, dtype=int).reshape(-1) - 1
    groups: list[np.ndarray] = []
    start = 0
    for size in sizes:
        end = start + int(size)
        groups.append(s[start:end].astype(int))
        start = end
    return groups


def _composition_eval(
    pop_dec: np.ndarray,
    o: np.ndarray,
    mat: np.ndarray,
    lambdas: np.ndarray,
    deltas: np.ndarray,
    biases: np.ndarray,
    funcs: list[Callable[[np.ndarray], np.ndarray]],
    constant_bias: float,
) -> np.ndarray:
    n, d = pop_dec.shape
    k = len(funcs)
    w = np.zeros((n, k), dtype=float)
    f = np.zeros((n, k), dtype=float)
    for i in range(k):
        shift = o[i, :d]
        diff = pop_dec - shift[None, :]
        tmp = np.sum(diff**2, axis=1)
        w[:, i] = 1.0 / (np.sqrt(tmp) + 1e-10) * np.exp(-tmp / (2.0 * d * deltas[i] ** 2))
        rot = mat[i * d : (i + 1) * d, :]
        f[:, i] = funcs[i](diff @ rot.T)
    denom = np.sum(w, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    w = w / denom
    return float(constant_bias) + np.sum(w * (lambdas[None, :] * f + biases[None, :]), axis=1)


class _BaseCEC2020(Problem):
    _USE_JAX = False
    _DATA_INDEX = 1

    def __init__(self, n_var: int | None = None, **kwargs):
        self._data = _cec2020_data()
        self._item = self._data[int(self._DATA_INDEX) - 1]
        d = _select_dim(n_var)
        self.D = d
        self.O = np.asarray(getattr(self._item, "o"), dtype=float)
        self.Mat = _get_field_dim(self._item, "M", d)
        xl = np.full(d, -100.0, dtype=float)
        xu = np.full(d, 100.0, dtype=float)
        super().__init__(n_var=d, n_obj=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _xp(self):
        return _xp(self._USE_JAX)

    def _prepare_np(self, x) -> np.ndarray:
        return _clip_bounds_np(x, np.asarray(self.xl, dtype=float), np.asarray(self.xu, dtype=float))

    def _out(self, out, f):
        out["F"] = np.asarray(f, dtype=float).reshape(-1, 1)


class CEC2020_F1(_BaseCEC2020):
    """Bent cigar function. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 1

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prepare_np(x)
        z = x - self.O[: x.shape[1]][None, :]
        y = z @ self.Mat.T
        rest = np.sum(y[:, 1:] ** 2, axis=1) if y.shape[1] > 1 else 0.0
        self._out(out, 100.0 + y[:, 0] ** 2 + 1e6 * rest)


class CEC2020_F2(_BaseCEC2020):
    """Shifted and rotated Schwefel function. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 2

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prepare_np(x)
        z = x - self.O[: x.shape[1]][None, :]
        y = 10.0 * (z @ self.Mat.T)
        zz = y + 4.2097e2
        g = zz * np.sin(np.sqrt(np.abs(zz)))
        mask_hi = zz > 500.0
        if np.any(mask_hi):
            temp = 500.0 - np.mod(zz[mask_hi], 500.0)
            g[mask_hi] = temp * np.sin(np.sqrt(np.abs(temp))) - (zz[mask_hi] - 500.0) ** 2 / 10000.0 / self.n_var
        mask_lo = zz < -500.0
        if np.any(mask_lo):
            temp = np.mod(np.abs(zz[mask_lo]), 500.0) - 500.0
            g[mask_lo] = temp * np.sin(np.sqrt(np.abs(temp))) - (zz[mask_lo] - 500.0) / 10000.0 / self.n_var
        f = 1100.0 + 418.9829 * self.n_var - np.sum(g, axis=1)
        self._out(out, f)


class CEC2020_F3(_BaseCEC2020):
    """Shifted and rotated Lunacek bi-Rastrigin. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 3

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prepare_np(x)
        s = 1.0 - 1.0 / (2.0 * math.sqrt(self.n_var + 20.0) - 8.2)
        mu0 = 2.5
        mu1 = -math.sqrt((mu0**2 - 1.0) / s)
        o = self.O[: x.shape[1]]
        y = (x - o[None, :]) / 10.0
        tmp = 2.0 * np.sign(o)[None, :] * y + mu0
        z = (tmp - mu0) @ self.Mat.T
        f = 700.0 + np.minimum(np.sum((tmp - mu0) ** 2, axis=1), self.n_var + s * np.sum((tmp - mu1) ** 2, axis=1)) + 10.0 * (self.n_var - np.sum(np.cos(2.0 * np.pi * z), axis=1))
        self._out(out, f)


class CEC2020_F4(_BaseCEC2020):
    """Expanded Rosenbrock plus Griewank. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 4

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prepare_np(x)
        z = x - self.O[: x.shape[1]][None, :]
        y = 0.05 * (z @ self.Mat.T)
        zz = y + 1.0
        nxt = zz[:, [*range(1, zz.shape[1]), 0]]
        temp = 100.0 * (zz**2 - nxt) ** 2 + (zz - 1.0) ** 2
        f = 1900.0 + np.sum(temp**2 / 4000.0 - np.cos(temp) + 1.0, axis=1)
        self._out(out, f)


class _BaseCEC2020Hybrid(_BaseCEC2020):
    _PARTITION: list[float] = []
    _PARTITION_D5: list[int] | None = None
    _FUNCS: list[Callable[[np.ndarray], np.ndarray]] = []
    _BIAS = 0.0

    def __init__(self, n_var: int | None = None, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        self.S_perm = _get_field_dim(self._item, "S", self.n_var).astype(int)
        if self._PARTITION_D5 is not None and self.n_var == 5:
            sizes = [int(v) for v in self._PARTITION_D5]
        else:
            sizes = np.ceil(np.asarray(self._PARTITION, dtype=float) * self.n_var).astype(int).tolist()
            if sizes:
                sizes[0] = int(self.n_var - sum(sizes[1:]))
        self.S = _partition_indices(self.S_perm, sizes)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prepare_np(x)
        y = x - self.O[: x.shape[1]][None, :]
        z = y @ self.Mat.T
        total = float(self._BIAS)
        f = np.full(x.shape[0], total, dtype=float)
        for group, func in zip(self.S, self._FUNCS):
            f += func(z[:, group])
        self._out(out, f)


class CEC2020_F5(_BaseCEC2020Hybrid):
    """Hybrid function 1. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 5
    _PARTITION = [0.3, 0.3, 0.4]
    _FUNCS = [_schwefel_shifted, _rastrigin, _elliptic]
    _BIAS = 1700.0


class CEC2020_F6(_BaseCEC2020Hybrid):
    """Hybrid function 2. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 6
    _PARTITION = [0.2, 0.2, 0.3, 0.3]
    _PARTITION_D5 = [1, 1, 1, 2]
    _FUNCS = [_schaffer, _hgbat, _rosenbrock_scaled, _schwefel_shifted]
    _BIAS = 1600.0


class CEC2020_F7(_BaseCEC2020Hybrid):
    """Hybrid function 3. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 7
    _PARTITION = [0.1, 0.2, 0.2, 0.2, 0.3]
    _PARTITION_D5 = [1, 1, 1, 1, 1]
    _FUNCS = [_schaffer, _hgbat, _rosenbrock_scaled, _schwefel_shifted, _elliptic]
    _BIAS = 2100.0


class _BaseCEC2020Composition(_BaseCEC2020):
    _LAMBDAS: list[float] = []
    _DELTAS: list[float] = []
    _BIASES: list[float] = []
    _FUNCS: list[Callable[[np.ndarray], np.ndarray]] = []
    _CONST_BIAS: float = 0.0

    def __init__(self, n_var: int | None = None, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        self.Om = np.asarray(self.O, dtype=float)
        self.Mstack = np.asarray(self.Mat, dtype=float)
        self.lambdas = np.asarray(self._LAMBDAS, dtype=float)
        self.deltas = np.asarray(self._DELTAS, dtype=float)
        self.biases = np.asarray(self._BIASES, dtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._prepare_np(x)
        f = _composition_eval(
            x,
            self.Om,
            self.Mstack,
            self.lambdas,
            self.deltas,
            self.biases,
            list(self._FUNCS),
            self._CONST_BIAS,
        )
        self._out(out, f)


class CEC2020_F8(_BaseCEC2020Composition):
    """Composition function 1. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 8
    _LAMBDAS = [1.0, 10.0, 1.0]
    _DELTAS = [10.0, 20.0, 30.0]
    _BIASES = [0.0, 100.0, 200.0]
    _FUNCS = [_rastrigin, _griewank, _schwefel_shifted]
    _CONST_BIAS = 2200.0


class CEC2020_F9(_BaseCEC2020Composition):
    """Composition function 2. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 9
    _LAMBDAS = [10.0, 1e-6, 10.0, 1.0]
    _DELTAS = [10.0, 20.0, 30.0, 40.0]
    _BIASES = [0.0, 100.0, 200.0, 300.0]
    _FUNCS = [_ackley, _elliptic, _griewank, _rastrigin]
    _CONST_BIAS = 2400.0


class CEC2020_F10(_BaseCEC2020Composition):
    """Composition function 3. Ref: Yue et al. (2019)."""

    _DATA_INDEX = 10
    _LAMBDAS = [10.0, 1.0, 10.0, 1e-6, 1.0]
    _DELTAS = [10.0, 20.0, 30.0, 40.0, 50.0]
    _BIASES = [0.0, 100.0, 200.0, 300.0, 400.0]
    _FUNCS = [_rastrigin, _happycat, _ackley, _discus, _rosenbrock_scaled]
    _CONST_BIAS = 2500.0


_CPU = [f"CEC2020_F{i}" for i in range(1, 11)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU + [f"{name}_JAX" for name in _CPU]

