"""MLX-accelerated WFG problem variants (Apple Silicon).

`_MLX` evaluation variants for the WFG problems whose transformation chain is
*functional* (no in-place slice mutation): WFG4 and WFG5. Each subclasses its CPU
sibling (identical dimensions, bounds, S/A vectors) and overrides ``_evaluate`` to
run the shift/reduction/shape chain on Apple's MLX framework. MLX is imported
lazily; when unavailable, evaluation falls back to the validated CPU
implementation, so results stay correct on any platform.

Scope note: WFG1/2/3/6/7/8/9 use in-place slice assignment, non-separable
reductions, or parameter-dependent loops that port poorly to MLX (immutable
arrays, little parallelism). They keep the CPU path; the JAX aliases in
``wfg.py`` are likewise empty CPU-delegating subclasses.
"""
from __future__ import annotations

import numpy as np

from problems.many.wfg import WFG4 as _CpuWFG4
from problems.many.wfg import WFG5 as _CpuWFG5

try:
    import mlx.core as mx

    _HAS_MLX = True
except Exception:  # noqa: BLE001
    mx = None  # type: ignore[assignment]
    _HAS_MLX = False

_PI = float(np.pi)


def _correct01(X, eps: float = 1.0e-10):
    X = mx.where((X < 0) & (X >= -eps), mx.zeros_like(X), X)
    X = mx.where((X > 1) & (X <= 1 + eps), mx.ones_like(X), X)
    return X


def _shift_multi_modal(y, A, B, C):
    tmp1 = mx.abs(y - C) / (2.0 * (mx.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * _PI * (0.5 - tmp1)
    ret = (1.0 + mx.cos(tmp2) + 4.0 * B * tmp1 ** 2) / (B + 2.0)
    return _correct01(ret)


def _param_deceptive(y, A=0.35, B=0.001, C=0.05):
    tmp1 = mx.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = mx.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (mx.abs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return _correct01(ret)


def _wsum_uniform(y):
    return _correct01(mx.mean(y, axis=1))


def _t2(x, m: int, k: int):
    """Uniform weighted-sum reduction over position blocks (WFG4/WFG5 t2)."""
    gap = k // (m - 1)
    cols = [_wsum_uniform(x[:, (i - 1) * gap: i * gap]) for i in range(1, m)]
    cols.append(_wsum_uniform(x[:, k:]))
    return mx.stack(cols, axis=1)


def _concave(x, m: int):
    M = x.shape[1]
    if m == 1:
        ret = mx.prod(mx.sin(0.5 * x[:, :M] * _PI), axis=1)
    elif 1 < m <= M:
        ret = mx.prod(mx.sin(0.5 * x[:, : M - m + 1] * _PI), axis=1)
        ret = ret * mx.cos(0.5 * x[:, M - m + 1] * _PI)
    else:
        ret = mx.cos(0.5 * x[:, 0] * _PI)
    return _correct01(ret)


def _post(t, a):
    last = t[:, -1]
    cols = []
    for i in range(t.shape[1] - 1):
        cols.append(mx.maximum(last, float(a[i])) * (t[:, i] - 0.5) + 0.5)
    cols.append(last)
    return mx.stack(cols, axis=1)


def _calculate(t, s, h):
    H = mx.stack(h, axis=1)
    s_mx = mx.array(np.asarray(s, dtype=np.float32))
    return mx.expand_dims(t[:, -1], 1) + s_mx * H


def _eval_wfg(problem, x, out, t1):
    xm = mx.array(np.atleast_2d(np.asarray(x, dtype=np.float32)))
    xu = mx.array(np.asarray(problem.xu, dtype=np.float32))
    y = xm / xu
    y = t1(y)
    y = _t2(y, problem.n_obj, problem.k)
    y = _post(y, problem.A)
    h = [_concave(y[:, :-1], m + 1) for m in range(problem.n_obj)]
    out["F"] = np.asarray(_calculate(y, problem.S, h), dtype=float)


class WFG4_MLX(_CpuWFG4):
    """MLX evaluation variant of WFG4."""

    def _evaluate(self, x, out, *args, **kwargs):
        if not _HAS_MLX:
            return super()._evaluate(x, out, *args, **kwargs)
        _eval_wfg(self, x, out, lambda y: _shift_multi_modal(y, 30.0, 10.0, 0.35))


class WFG5_MLX(_CpuWFG5):
    """MLX evaluation variant of WFG5."""

    def _evaluate(self, x, out, *args, **kwargs):
        if not _HAS_MLX:
            return super()._evaluate(x, out, *args, **kwargs)
        _eval_wfg(self, x, out, lambda y: _param_deceptive(y, A=0.35, B=0.001, C=0.05))


__all__ = ["WFG4_MLX", "WFG5_MLX"]
