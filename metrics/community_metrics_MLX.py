"""MLX-accelerated community metrics (Apple Silicon).

Mirror of ``community_metrics_JAX.py`` using Apple's MLX framework. Covers the
compute-heavy distance metrics GD and IGD natively on MLX; HV delegates to the
CPU implementation (as the JAX variant also does). MLX is imported lazily and,
when unavailable, every metric falls back to the NumPy CPU sibling so results
stay correct on any platform.

MLX specifics: arrays default to float32 (Metal-friendly), are realized at the
boundary via ``np.asarray`` (which forces evaluation), and the CPU module is
reused for context extraction and references.
"""
from typing import Any

import numpy as np

from . import community_metrics as _cpu

try:
    import mlx.core as mx

    _HAS_MLX = True
except Exception:  # noqa: BLE001
    mx = None  # type: ignore[assignment]
    _HAS_MLX = False


METRIC_REFERENCES = {f"{k}_MLX": v for k, v in _cpu.METRIC_REFERENCES.items()}


def _to_mx(values: Any):
    return mx.array(np.atleast_2d(np.asarray(values, dtype=np.float32)))


def _pairwise_euclidean_mlx(a: Any, b: Any):
    aa = _to_mx(a)
    bb = _to_mx(b)
    # diff[i, j, :] = aa[i] - bb[j]
    diff = mx.expand_dims(aa, 1) - mx.expand_dims(bb, 0)
    return mx.sqrt(mx.sum(diff * diff, axis=2))


def _scalar(value: Any) -> float:
    """Realize an MLX scalar to a Python float (forces evaluation)."""
    return float(np.asarray(value))


def _metric_GD_MLX(front, context):
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None:
        return float("nan")
    if not _HAS_MLX:
        return _cpu._gd_value(pop, opt)
    if pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    d = _pairwise_euclidean_mlx(pop, opt)
    nearest = mx.min(d, axis=1)
    n = max(1, int(nearest.shape[0]))
    return _scalar(mx.sqrt(mx.sum(nearest * nearest))) / n


def _metric_IGD_MLX(front, context):
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None:
        return float("nan")
    if not _HAS_MLX:
        return _cpu._igd_value(pop, opt)
    if pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    d = _pairwise_euclidean_mlx(opt, pop)
    return _scalar(mx.mean(mx.min(d, axis=1)))


def _metric_HV_MLX(front, context):
    # Hypervolume is delegated to the validated CPU implementation (the JAX
    # variant does the same); MLX acceleration targets the distance metrics.
    pop_obj = _cpu._get_front(front)
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _cpu._community_hv(pop_obj, optimum, context)


METRICS = {
    "GD_MLX": _metric_GD_MLX,
    "HV_MLX": _metric_HV_MLX,
    "IGD_MLX": _metric_IGD_MLX,
}
