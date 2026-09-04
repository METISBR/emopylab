from __future__ import annotations

import math
from typing import Any

import numpy as np
from core.problem import Problem

try:
    import jax.numpy as jnp
except Exception:  # noqa: BLE001
    jnp = None


_BBOB_2009_REF = (
    "N. Hansen, S. Finck, R. Ros, and A. Auger. Real-parameter black-box "
    "optimization benchmarking 2009: Noiseless functions definitions. "
    "RR-6829, INRIA, 2009."
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


def _lin_idx(d: int, xp):
    if d <= 1:
        return xp.zeros((int(d),), dtype=float)
    return xp.arange(0, int(d), dtype=float) / float(d - 1)


def _diag_power(base: float, coeff: float, d: int, xp):
    idx = _lin_idx(d, xp)
    return xp.power(float(base), float(coeff) * idx)


def _diag_matrix(vals: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(vals, dtype=float).reshape(-1))


def _seed_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def _orth_from_seed(seed: int, d: int) -> np.ndarray:
    rng = _seed_rng(seed)
    a = rng.rand(int(d), int(d))
    q, r = np.linalg.qr(a)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    q = q * s[np.newaxis, :]
    return np.asarray(q, dtype=float)


def _rand_sign(seed: int, d: int, scale: float) -> np.ndarray:
    rng = _seed_rng(seed)
    return float(scale) * np.sign(rng.rand(1, int(d)).reshape(-1) - 0.5)


def _parse_xopt(n_var: int | None, xopt: float | list[float] | np.ndarray) -> tuple[int, np.ndarray]:
    if np.isscalar(xopt):
        d = 30 if n_var is None else max(1, int(n_var))
        vec = np.full(d, float(xopt), dtype=float)
        return d, vec
    vec = np.asarray(xopt, dtype=float).reshape(-1)
    if vec.size == 0:
        d = 30 if n_var is None else max(1, int(n_var))
        vec = np.zeros(d, dtype=float)
        return d, vec
    return int(vec.size), vec.astype(float)


def _fpen(x, xp):
    return xp.sum(xp.maximum(0.0, xp.abs(x) - 5.0) ** 2, axis=1)


def _tosz(x, xp):
    x = xp.asarray(x, dtype=float)
    x_abs = xp.abs(x)
    xh = xp.where(x != 0, xp.log(x_abs), 0.0)
    c1 = xp.where(x > 0, 10.0, 5.5)
    c2 = xp.where(x > 0, 7.9, 3.1)
    return xp.sign(x) * xp.exp(xh + 0.049 * (xp.sin(c1 * xh) + xp.sin(c2 * xh)))


def _tasy(beta: float, x, xp):
    x = xp.asarray(x, dtype=float)
    d = int(x.shape[1])
    idx = _lin_idx(d, xp)
    sqrt_pos = xp.sqrt(xp.maximum(x, 0.0))
    exponent = 1.0 + float(beta) * idx[xp.newaxis, :] * sqrt_pos
    x2 = xp.power(xp.maximum(x, 0.0), exponent)
    return xp.where(x > 0, x2, x)


def _rosenbrock_sum(z, xp):
    if z.shape[1] <= 1:
        return xp.zeros((z.shape[0],), dtype=float)
    return xp.sum(100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1.0) ** 2, axis=1)


def _mat(xp, arr: np.ndarray):
    return xp.asarray(np.asarray(arr, dtype=float), dtype=float)


class _BaseBBOB(Problem):
    _USE_JAX = False

    def _xp(self):
        return _xp(self._USE_JAX)

    def _arr(self, x):
        return _as_2d(x, self._xp())

    def _out(self, out, f):
        out["F"] = _to_numpy(f).reshape(-1, 1)

    @staticmethod
    def _bounds(d: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(int(d), -5.0, dtype=float), np.full(int(d), 5.0, dtype=float)


class _BaseBBOBXopt(_BaseBBOB):
    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        d, xopt_vec = _parse_xopt(n_var, xopt)
        self.xopt = xopt_vec
        xl, xu = self._bounds(d)
        super().__init__(n_var=d, n_obj=1, xl=xl, xu=xu, vtype=float, **kwargs)


class BBOB_F1(_BaseBBOBXopt):
    """Sphere function. Ref: Hansen et al. (2009)."""

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = x - _mat(xp, self.xopt)[xp.newaxis, :]
        self._out(out, xp.sum(z**2, axis=1))


class BBOB_F2(_BaseBBOBXopt):
    """Ellipsoidal function. Ref: Hansen et al. (2009)."""

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = _tosz(x - _mat(xp, self.xopt)[xp.newaxis, :], xp)
        w = _diag_power(10.0, 6.0, self.n_var, xp)
        self._out(out, xp.sum(w[xp.newaxis, :] * z**2, axis=1))


class BBOB_F3(_BaseBBOBXopt):
    """Rastrigin function. Ref: Hansen et al. (2009)."""

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = _tosz(x - _mat(xp, self.xopt)[xp.newaxis, :], xp)
        z = _tasy(0.2, z, xp)
        s = _diag_power(10.0, 0.5, self.n_var, xp)
        z = s[xp.newaxis, :] * z
        f = 10.0 * (float(self.n_var) - xp.sum(xp.cos(2.0 * xp.pi * z), axis=1)) + xp.sum(z**2, axis=1)
        self._out(out, f)


class BBOB_F4(_BaseBBOBXopt):
    """Buche-Rastrigin function. Ref: Hansen et al. (2009)."""

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        x_raw = x
        z = _tosz(x - _mat(xp, self.xopt)[xp.newaxis, :], xp)
        s = _diag_power(10.0, 0.5, self.n_var, xp)
        s = xp.tile(s[xp.newaxis, :], (x.shape[0], 1))
        odd = np.arange(int(self.n_var)) % 2 == 0
        odd_mask = xp.asarray(odd[np.newaxis, :])
        s = xp.where(odd_mask & (z > 0), 10.0 * s, s)
        z = s * z
        f = 10.0 * (float(self.n_var) - xp.sum(xp.cos(2.0 * xp.pi * z), axis=1)) + xp.sum(z**2, axis=1) + 100.0 * _fpen(x_raw, xp)
        self._out(out, f)


class BBOB_F5(_BaseBBOB):
    """Linear slope function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int = 30, **kwargs):
        d = max(1, int(n_var))
        self.xopt = _rand_sign(1, d, 5.0)
        xl, xu = self._bounds(d)
        super().__init__(n_var=d, n_obj=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = x
        xopt = _mat(xp, self.xopt)[xp.newaxis, :]
        z = xp.where(xopt * x >= 25.0, xopt, z)
        s = xp.sign(xopt) * _diag_power(10.0, 1.0, self.n_var, xp)[xp.newaxis, :]
        f = xp.sum(5.0 * xp.abs(s) - s * z, axis=1)
        self._out(out, f)


class BBOB_F6(_BaseBBOBXopt):
    """Attractive sector function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.Q = _orth_from_seed(1, self.n_var)
        self.R = _orth_from_seed(2, self.n_var)
        self.QDR = self.Q @ _diag_matrix(_diag_power(10.0, 0.5, self.n_var, np)) @ self.R

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        xopt = _mat(xp, self.xopt)[xp.newaxis, :]
        z = (x - xopt) @ _mat(xp, self.QDR)
        s = xp.ones_like(z)
        s = xp.where(z * xopt > 0, 100.0, s)
        f = xp.power(_tosz(xp.sum((s * z) ** 2, axis=1), xp), 0.9)
        self._out(out, f)


class BBOB_F7(_BaseBBOBXopt):
    """Step ellipsoidal function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.Q = _orth_from_seed(1, self.n_var)
        self.R = _orth_from_seed(2, self.n_var)
        self.DR = _diag_matrix(_diag_power(10.0, 0.5, self.n_var, np)) @ self.R
        self.w2 = _diag_power(10.0, 2.0, self.n_var, np)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        x_raw = x
        xopt = _mat(xp, self.xopt)[xp.newaxis, :]
        z = (x - xopt) @ _mat(xp, self.DR)
        zh = xp.floor(0.5 + 10.0 * z) / 10.0
        zh = xp.where(xp.abs(z) > 0.5, xp.floor(0.5 + z), zh)
        term1 = xp.abs(zh[:, 0]) / 1e4
        zq = z @ _mat(xp, self.Q)
        term2 = xp.sum(_mat(xp, self.w2)[xp.newaxis, :] * zq**2, axis=1)
        f = 0.1 * xp.maximum(term1, term2) + _fpen(x_raw, xp)
        self._out(out, f)


class BBOB_F8(_BaseBBOBXopt):
    """Rosenbrock function. Ref: Hansen et al. (2009)."""

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        c = max(1.0, math.sqrt(float(self.n_var)) / 8.0)
        z = c * (x - _mat(xp, self.xopt)[xp.newaxis, :]) + 1.0
        self._out(out, _rosenbrock_sum(z, xp))


class BBOB_F9(_BaseBBOBXopt):
    """Rotated Rosenbrock function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.R = _orth_from_seed(2, self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        c = max(1.0, math.sqrt(float(self.n_var)) / 8.0)
        z = c * ((x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R)) + 0.5
        self._out(out, _rosenbrock_sum(z, xp))


class BBOB_F10(_BaseBBOBXopt):
    """Rotated ellipsoidal function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.R = _orth_from_seed(2, self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = _tosz((x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R), xp)
        w = _diag_power(10.0, 6.0, self.n_var, xp)
        self._out(out, xp.sum(w[xp.newaxis, :] * z**2, axis=1))


class BBOB_F11(_BaseBBOBXopt):
    """Discus function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.R = _orth_from_seed(2, self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = _tosz((x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R), xp)
        rest = xp.sum(z[:, 1:] ** 2, axis=1) if self.n_var > 1 else 0.0
        self._out(out, 1e6 * z[:, 0] ** 2 + rest)


class BBOB_F12(_BaseBBOBXopt):
    """Bent cigar function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.R = _orth_from_seed(2, self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        r = _mat(xp, self.R)
        z = _tasy(0.5, (x - _mat(xp, self.xopt)[xp.newaxis, :]) @ r, xp) @ r
        rest = xp.sum(z[:, 1:] ** 2, axis=1) if self.n_var > 1 else 0.0
        self._out(out, z[:, 0] ** 2 + 1e6 * rest)


class BBOB_F13(_BaseBBOBXopt):
    """Sharp ridge function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.Q = _orth_from_seed(1, self.n_var)
        self.R = _orth_from_seed(2, self.n_var)
        self.QDR = self.Q @ _diag_matrix(_diag_power(10.0, 0.5, self.n_var, np)) @ self.R

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = (x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.QDR)
        rest = xp.sum(z[:, 1:] ** 2, axis=1) if self.n_var > 1 else 0.0
        self._out(out, z[:, 0] ** 2 + 100.0 * xp.sqrt(rest))


class BBOB_F14(_BaseBBOBXopt):
    """Different powers function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.R = _orth_from_seed(2, self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = (x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R)
        expo = 2.0 + 4.0 * _lin_idx(self.n_var, xp)
        f = xp.sqrt(xp.sum(xp.abs(z) ** expo[xp.newaxis, :], axis=1))
        self._out(out, f)


class BBOB_F15(_BaseBBOBXopt):
    """Rotated Rastrigin function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.Q = _orth_from_seed(1, self.n_var)
        self.R = _orth_from_seed(2, self.n_var)
        self.RDQ = self.R @ _diag_matrix(_diag_power(10.0, 0.5, self.n_var, np)) @ self.Q

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        z = _tosz((x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R), xp)
        z = _tasy(0.2, z, xp) @ _mat(xp, self.RDQ)
        f = 10.0 * (float(self.n_var) - xp.sum(xp.cos(2.0 * xp.pi * z), axis=1)) + xp.sum(z**2, axis=1)
        self._out(out, f)


class BBOB_F16(_BaseBBOBXopt):
    """Weierstrass function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.Q = _orth_from_seed(1, self.n_var)
        self.R = _orth_from_seed(2, self.n_var)
        self.RDQ = self.R @ _diag_matrix(_diag_power(0.01, 0.5, self.n_var, np)) @ self.Q
        k = np.arange(0, 12, dtype=float)
        self._wk = (0.5**k, 3.0**k)
        self._f0 = float(np.sum((0.5**k) * np.cos(np.pi * (3.0**k))))

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        x_raw = x
        z = _tosz((x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R), xp)
        z = z @ _mat(xp, self.RDQ)
        ak = _mat(xp, self._wk[0])
        bk = _mat(xp, self._wk[1])
        zz = z[:, :, xp.newaxis] + 0.5
        inner = xp.sum(ak[xp.newaxis, xp.newaxis, :] * xp.cos(2.0 * xp.pi * bk[xp.newaxis, xp.newaxis, :] * zz), axis=2)
        mean_inner = xp.mean(inner, axis=1)
        f = 10.0 * (mean_inner - float(self._f0)) ** 3 + 10.0 / float(self.n_var) * _fpen(x_raw, xp)
        self._out(out, f)


class BBOB_F17(_BaseBBOBXopt):
    """Schaffers F7 function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.Q = _orth_from_seed(1, self.n_var)
        self.R = _orth_from_seed(2, self.n_var)
        self.DQ = _diag_matrix(_diag_power(10.0, 0.5, self.n_var, np)) @ self.Q

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        x_raw = x
        z = _tasy(0.5, (x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R), xp) @ _mat(xp, self.DQ)
        if self.n_var <= 1:
            s = xp.abs(z[:, :1])
        else:
            s = xp.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)
        g = xp.sqrt(s) + xp.sqrt(s) * xp.sin(50.0 * s**0.2) ** 2
        f = xp.mean(g, axis=1) ** 2 + 10.0 * _fpen(x_raw, xp)
        self._out(out, f)


class BBOB_F18(_BaseBBOBXopt):
    """Moderately ill-conditioned Schaffers F7. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.Q = _orth_from_seed(1, self.n_var)
        self.R = _orth_from_seed(2, self.n_var)
        self.DQ = _diag_matrix(_diag_power(1000.0, 0.5, self.n_var, np)) @ self.Q

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        x_raw = x
        z = _tasy(0.5, (x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R), xp) @ _mat(xp, self.DQ)
        if self.n_var <= 1:
            s = xp.abs(z[:, :1])
        else:
            s = xp.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)
        g = xp.sqrt(s) + xp.sqrt(s) * xp.sin(50.0 * s**0.2) ** 2
        f = xp.mean(g, axis=1) ** 2 + 10.0 * _fpen(x_raw, xp)
        self._out(out, f)


class BBOB_F19(_BaseBBOBXopt):
    """Composite Griewank-Rosenbrock F8F2. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.R = _orth_from_seed(2, self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        c = max(1.0, math.sqrt(float(self.n_var)) / 8.0)
        z = c * ((x - _mat(xp, self.xopt)[xp.newaxis, :]) @ _mat(xp, self.R)) + 0.5
        if self.n_var <= 1:
            s = xp.zeros((x.shape[0], 1), dtype=float)
        else:
            s = 100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1.0) ** 2
        f = 10.0 * xp.mean(s / 4000.0 - xp.cos(s), axis=1) + 10.0
        self._out(out, f)


class BBOB_F20(_BaseBBOB):
    """Schwefel function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int = 30, **kwargs):
        d = max(1, int(n_var))
        self.xopt = _rand_sign(1, d, 4.2096874633 / 2.0)
        self.scale = _diag_power(10.0, 0.5, d, np)
        xl, xu = self._bounds(d)
        super().__init__(n_var=d, n_obj=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        xopt = _mat(xp, self.xopt)[xp.newaxis, :]
        xx = 2.0 * xp.sign(xopt) * x
        if xp is not np:
            # jax-compatible sequential recurrence fallback with numpy for simplicity.
            xx_np = _to_numpy(xx)
            xopt_np = _to_numpy(xopt)
            z_np = np.zeros_like(xx_np)
            z_np[:, 0] = xx_np[:, 0]
            if self.n_var > 1:
                z_np[:, 1:] = xx_np[:, 1:] + 0.25 * (xx_np[:, :-1] - 2.0 * np.abs(xopt_np[:, :-1]))
            zz = 100.0 * (((z_np - 2.0 * np.abs(xopt_np)) * self.scale[np.newaxis, :]) + 2.0 * np.abs(xopt_np))
            f = -0.01 * np.mean(zz * np.sin(np.sqrt(np.abs(zz))), axis=1) + 4.189828872724339 + 100.0 * np.sum(np.maximum(0.0, np.abs(zz / 100.0) - 5.0) ** 2, axis=1)
            out["F"] = f.reshape(-1, 1)
            return
        z = xp.zeros_like(xx)
        z[:, 0] = xx[:, 0]
        if self.n_var > 1:
            z[:, 1:] = xx[:, 1:] + 0.25 * (xx[:, :-1] - 2.0 * xp.abs(xopt[:, :-1]))
        zz = 100.0 * (((z - 2.0 * xp.abs(xopt)) * _mat(xp, self.scale)[xp.newaxis, :]) + 2.0 * xp.abs(xopt))
        f = -0.01 * xp.mean(zz * xp.sin(xp.sqrt(xp.abs(zz))), axis=1) + 4.189828872724339 + 100.0 * _fpen(zz / 100.0, xp)
        self._out(out, f)


class _BaseGallagher(_BaseBBOB):
    _N_PEAKS = 101
    _FIRST_ALPHA = 1000.0
    _Y0_SPAN = 8.0
    _Y0_SHIFT = -4.0
    _YR_SPAN = 10.0
    _YR_SHIFT = -5.0

    def __init__(self, n_var: int = 30, **kwargs):
        d = max(1, int(n_var))
        self.xopt = _rand_sign(1, d, 4.2096874633 / 2.0)
        self.R = _orth_from_seed(2, d)
        alpha_tail = np.power(1000.0, 2.0 * np.arange(0, self._N_PEAKS - 1, dtype=float) / max(1, self._N_PEAKS - 2))
        perm = _seed_rng(2).permutation(alpha_tail.size)
        alpha = np.concatenate([[self._FIRST_ALPHA], alpha_tail[perm]])
        self.C = []
        self.A = []
        idx = np.arange(0, d, dtype=float) / max(1, d - 1)
        for a in alpha:
            c = np.diag(np.power(a, 0.5 * idx)) / (a**0.25)
            self.C.append(c)
            self.A.append(self.R.T @ c @ self.R)
        y0 = _seed_rng(3).rand() * self._Y0_SPAN + self._Y0_SHIFT
        yr = _seed_rng(4).rand(self._N_PEAKS - 1) * self._YR_SPAN + self._YR_SHIFT
        self.y = np.concatenate([[y0], yr]).astype(float)
        self.w = np.concatenate([[10.0], 1.1 + 8.0 * np.arange(0, self._N_PEAKS - 1, dtype=float) / max(1, self._N_PEAKS - 2)])
        xl, xu = self._bounds(d)
        super().__init__(n_var=d, n_obj=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x_np = _to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        temp = np.zeros((x_np.shape[0], self._N_PEAKS), dtype=float)
        for i in range(self._N_PEAKS):
            dx = x_np - self.y[i]
            quad = np.mean((dx @ self.A[i]) * dx, axis=1)
            temp[:, i] = self.w[i] * np.exp(-0.5 * quad)
        f = _to_numpy(_tosz(10.0 - np.max(temp, axis=1), np)) ** 2 + np.sum(np.maximum(0.0, np.abs(x_np) - 5.0) ** 2, axis=1)
        out["F"] = f.reshape(-1, 1)


class BBOB_F21(_BaseGallagher):
    """Gallagher's Gaussian 101-me peaks. Ref: Hansen et al. (2009)."""

    _N_PEAKS = 101
    _FIRST_ALPHA = 1000.0
    _Y0_SPAN = 8.0
    _Y0_SHIFT = -4.0
    _YR_SPAN = 10.0
    _YR_SHIFT = -5.0


class BBOB_F22(_BaseGallagher):
    """Gallagher's Gaussian 21-hi peaks. Ref: Hansen et al. (2009)."""

    _N_PEAKS = 21
    _FIRST_ALPHA = 1e6
    _Y0_SPAN = 7.84
    _Y0_SHIFT = -3.92
    _YR_SPAN = 9.8
    _YR_SHIFT = -4.9


class BBOB_F23(_BaseBBOBXopt):
    """Katsuura function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int | None = None, xopt: float | list[float] | np.ndarray = 0.0, **kwargs):
        super().__init__(n_var=n_var, xopt=xopt, **kwargs)
        self.Q = _orth_from_seed(1, self.n_var)
        self.R = _orth_from_seed(2, self.n_var)
        self.QDR = self.Q @ _diag_matrix(_diag_power(100.0, 0.5, self.n_var, np)) @ self.R
        self._J = np.power(2.0, np.arange(1, 33, dtype=float))
        self._i = np.arange(1, self.n_var + 1, dtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        x_np = _to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        z = (x_np - self.xopt[np.newaxis, :]) @ self.QDR
        zz = z[:, :, None]
        j = self._J[None, None, :]
        frac = np.abs(j * zz - np.round(j * zz)) / j
        inner = np.sum(frac, axis=2)
        term = np.power(1.0 + self._i[None, :] * inner, 10.0 / (float(self.n_var) ** 1.2))
        f = 10.0 / (float(self.n_var) ** 2) * np.prod(term, axis=1) - 10.0 / (float(self.n_var) ** 2) + np.sum(np.maximum(0.0, np.abs(x_np) - 5.0) ** 2, axis=1)
        out["F"] = f.reshape(-1, 1)


class BBOB_F24(_BaseBBOB):
    """Lunacek bi-Rastrigin function. Ref: Hansen et al. (2009)."""

    def __init__(self, n_var: int = 30, **kwargs):
        d = max(1, int(n_var))
        self.xopt = 1.25 * np.sign(_seed_rng(1).rand(1, d).reshape(-1) - 0.5)
        self.Q = _orth_from_seed(1, d)
        self.R = _orth_from_seed(2, d)
        self.QDR = self.Q @ _diag_matrix(_diag_power(100.0, 0.5, d, np)) @ self.R
        xl, xu = self._bounds(d)
        super().__init__(n_var=d, n_obj=1, xl=xl, xu=xu, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xp = self._xp()
        x = self._arr(x)
        x0 = 2.0 * xp.sign(_mat(xp, self.xopt)[xp.newaxis, :]) * x
        z = (x0 - 2.5) @ _mat(xp, self.QDR)
        s = 1.0 - 1.0 / (2.0 * math.sqrt(float(self.n_var) + 20.0) - 8.2)
        c1 = xp.sum((x0 - 2.5) ** 2, axis=1)
        c2 = float(self.n_var) + s * xp.sum((x0 + math.sqrt(5.25 / s)) ** 2, axis=1)
        f = xp.minimum(c1, c2) + 10.0 * (float(self.n_var) - xp.sum(xp.cos(2.0 * xp.pi * z), axis=1)) + 1e4 * _fpen(x0, xp)
        self._out(out, f)


_CPU = [f"BBOB_F{i}" for i in range(1, 25)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU + [f"{name}_JAX" for name in _CPU]
