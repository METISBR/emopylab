"""MLX-accelerated DTLZ problem variants (Apple Silicon).

`_MLX` evaluation variants of the compute-heavy DTLZ1/DTLZ2 problems. Each
variant subclasses its CPU sibling (identical bounds, dimensions, and
construction) and overrides ``_evaluate`` to run the objective math on Apple's
MLX framework. MLX is imported lazily; when unavailable, evaluation falls back
to the validated CPU implementation, so results stay correct on any platform.
"""
from __future__ import annotations

import numpy as np

from problems.many.dtlz import DTLZ1 as _CpuDTLZ1
from problems.many.dtlz import DTLZ2 as _CpuDTLZ2

try:
    import mlx.core as mx

    _HAS_MLX = True
except Exception:  # noqa: BLE001
    mx = None  # type: ignore[assignment]
    _HAS_MLX = False

_HALF_PI = float(np.pi / 2.0)


def _to_mx(x):
    return mx.clip(mx.array(np.atleast_2d(np.asarray(x, dtype=np.float32))), 0.0, 1.0)


def _obj_dtlz1_mlx(x, m: int, g):
    n = x.shape[0]
    ones = mx.ones((n, 1))
    left = mx.cumprod(mx.concatenate([ones, x[:, : m - 1]], axis=1), axis=1)[:, ::-1]
    right = mx.concatenate([ones, 1.0 - x[:, m - 2 :: -1]], axis=1)
    return 0.5 * mx.expand_dims(1.0 + g, 1) * left * right


def _obj_dtlz2_like_mlx(x, m: int, g):
    n = x.shape[0]
    cosp = mx.cos(x[:, : m - 1] * _HALF_PI)
    left = mx.cumprod(mx.concatenate([mx.ones((n, 1)), cosp], axis=1), axis=1)[:, ::-1]
    sinp = mx.sin(x[:, m - 2 :: -1] * _HALF_PI)
    right = mx.concatenate([mx.ones((n, 1)), sinp], axis=1)
    return mx.expand_dims(1.0 + g, 1) * left * right


class DTLZ1_MLX(_CpuDTLZ1):
    """MLX evaluation variant of DTLZ1."""

    def _evaluate(self, x, out, *args, **kwargs):
        if not _HAS_MLX:
            return super()._evaluate(x, out, *args, **kwargs)
        xm = _to_mx(x)
        tail = xm[:, self.n_obj - 1 :]
        g = 100.0 * (
            (self.n_var - self.n_obj + 1)
            + mx.sum((tail - 0.5) ** 2 - mx.cos(20.0 * mx.pi * (tail - 0.5)), axis=1)
        )
        out["F"] = np.asarray(_obj_dtlz1_mlx(xm, self.n_obj, g), dtype=float)


class DTLZ2_MLX(_CpuDTLZ2):
    """MLX evaluation variant of DTLZ2."""

    def _evaluate(self, x, out, *args, **kwargs):
        if not _HAS_MLX:
            return super()._evaluate(x, out, *args, **kwargs)
        xm = _to_mx(x)
        g = mx.sum((xm[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        out["F"] = np.asarray(_obj_dtlz2_like_mlx(xm, self.n_obj, g), dtype=float)


__all__ = ["DTLZ1_MLX", "DTLZ2_MLX"]
