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
    if "mlx" in type(values).__module__:
        if len(values.shape) == 1:
            return mx.expand_dims(values, 0)
        return values
    arr = np.atleast_2d(np.asarray(values, dtype=np.float32))
    return mx.array(arr)


if _HAS_MLX:
    @mx.compile
    def _pairwise_euclidean_mlx_compiled(aa: Any, bb: Any):
        diff = mx.expand_dims(aa, 1) - mx.expand_dims(bb, 0)
        return mx.sqrt(mx.sum(diff * diff, axis=2))

    @mx.compile
    def _gd_kernel_mlx(pop: Any, opt: Any):
        diff = mx.expand_dims(pop, 1) - mx.expand_dims(opt, 0)
        d = mx.sqrt(mx.sum(diff * diff, axis=2))
        nearest = mx.min(d, axis=1)
        return mx.sqrt(mx.sum(nearest * nearest)) / pop.shape[0]

    @mx.compile
    def _igd_kernel_mlx(pop: Any, opt: Any):
        diff = mx.expand_dims(opt, 1) - mx.expand_dims(pop, 0)
        d = mx.sqrt(mx.sum(diff * diff, axis=2))
        return mx.mean(mx.min(d, axis=1))

    @mx.compile
    def _igd_plus_kernel_mlx(pop: Any, opt: Any):
        # Modified distance: max(pop - opt, 0)
        diff = mx.maximum(mx.expand_dims(pop, 0) - mx.expand_dims(opt, 1), 0.0)
        d = mx.sqrt(mx.sum(diff * diff, axis=2))
        return mx.mean(mx.min(d, axis=1))

    @mx.compile
    def _spacing_kernel_mlx(pop: Any):
        diff = mx.abs(mx.expand_dims(pop, 1) - mx.expand_dims(pop, 0))
        d = mx.sum(diff, axis=2)
        # Mask diagonal
        eye = mx.eye(pop.shape[0]) * 1e9
        d_masked = d + eye
        min_d = mx.min(d_masked, axis=1)
        d_bar = mx.mean(min_d)
        dev = min_d - d_bar
        return mx.sqrt(mx.sum(dev * dev) / pop.shape[0])


def _scalar(value: Any) -> float:
    """Realize an MLX scalar to a Python float."""
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
    pop_m = _to_mx(pop)
    opt_m = _to_mx(opt)
    val = _gd_kernel_mlx(pop_m, opt_m)
    mx.eval(val)
    return _scalar(val)


def _metric_IGD_MLX(front, context):
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None:
        return float("nan")
    if not _HAS_MLX:
        return _cpu._igd_value(pop, opt)
    if pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    pop_m = _to_mx(pop)
    opt_m = _to_mx(opt)
    val = _igd_kernel_mlx(pop_m, opt_m)
    mx.eval(val)
    return _scalar(val)


def _metric_IGDp_MLX(front, context):
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None:
        return float("nan")
    if not _HAS_MLX:
        return _cpu._metric_IGDp(pop, context)
    if pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    pop_m = _to_mx(pop)
    opt_m = _to_mx(opt)
    val = _igd_plus_kernel_mlx(pop_m, opt_m)
    mx.eval(val)
    return _scalar(val)


def _metric_Spacing_MLX(front, context=None):
    pop = _cpu._get_front(front)
    if pop.size == 0 or pop.shape[0] < 2:
        return float("nan")
    if not _HAS_MLX:
        return _cpu._metric_Spacing(pop, context)
    pop_m = _to_mx(pop)
    val = _spacing_kernel_mlx(pop_m)
    mx.eval(val)
    return _scalar(val)


def _metric_DeltaP_MLX(front, context):
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None:
        return float("nan")
    if not _HAS_MLX:
        return _cpu._deltap_value(pop, opt)
    if pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    pop_m = _to_mx(pop)
    opt_m = _to_mx(opt)
    gd = _gd_kernel_mlx(pop_m, opt_m)
    igd = _igd_kernel_mlx(pop_m, opt_m)
    val = mx.maximum(gd, igd)
    mx.eval(val)
    return _scalar(val)


def _metric_HV_MLX(front, context):
    pop_obj = _cpu._get_front(front)
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _cpu._community_hv(pop_obj, optimum, context)


METRICS = {
    "GD_MLX": _metric_GD_MLX,
    "HV_MLX": _metric_HV_MLX,
    "IGD_MLX": _metric_IGD_MLX,
    "IGDp_MLX": _metric_IGDp_MLX,
    "Spacing_MLX": _metric_Spacing_MLX,
    "DeltaP_MLX": _metric_DeltaP_MLX,
}
