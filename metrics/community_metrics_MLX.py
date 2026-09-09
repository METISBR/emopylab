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


_TILE_BUDGET = 2 ** 26


def _tiled_min_mlx(aa: Any, bb: Any) -> Any:
    """Row-wise min Euclidean distance over query rows aa vs ref rows bb."""
    n, k = int(aa.shape[0]), int(bb.shape[0])
    m = int(aa.shape[1])
    tile = max(1, min(k, _TILE_BUDGET // max(1, n * m)))

    @mx.compile
    def _tile_kernel(x: Any, y: Any):
        diff = mx.expand_dims(x, 1) - mx.expand_dims(y, 0)
        d = mx.sqrt(mx.sum(diff * diff, axis=2))
        return mx.min(d, axis=1)

    best = None
    for start in range(0, k, tile):
        cur = _tile_kernel(aa, bb[start:start + tile])
        mx.eval(cur)
        best = cur if best is None else mx.minimum(best, cur)
    mx.eval(best)
    return best


def _tiled_min_igdp_mlx(pop_m: Any, opt_m: Any, p: float = 2.0) -> Any:
    """IGD+ row minima per OPT row, tiled over POP rows with context p-norm."""
    n, k = int(pop_m.shape[0]), int(opt_m.shape[0])
    m = int(pop_m.shape[1])
    tile = max(1, min(n, _TILE_BUDGET // max(1, k * m)))
    pf = float(p)

    @mx.compile
    def _tile_kernel_igdp(o: Any, chunk: Any):
        diff = mx.maximum(mx.expand_dims(chunk, 0) - mx.expand_dims(o, 1), 0.0)
        if abs(pf - 2.0) <= 1e-12:
            d = mx.sqrt(mx.sum(diff * diff, axis=2))
        elif abs(pf - 1.0) <= 1e-12:
            d = mx.sum(diff, axis=2)
        else:
            d = mx.sum(mx.power(diff, pf), axis=2) ** (1.0 / pf)
        return mx.min(d, axis=1)

    best = None
    for start in range(0, n, tile):
        cur = _tile_kernel_igdp(opt_m, pop_m[start:start + tile])
        mx.eval(cur)
        best = cur if best is None else mx.minimum(best, cur)
    mx.eval(best)
    return best


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
        # Mask diagonal with +inf while preserving off-diagonal distances.
        d_masked = d + mx.eye(pop.shape[0]) * float("inf")
        min_d = mx.min(d_masked, axis=1)
        n = float(pop.shape[0])
        d_bar = mx.mean(min_d)
        dev = min_d - d_bar
        return mx.sqrt(mx.sum(dev * dev) / max(1.0, n - 1.0))


def _scalar(value: Any) -> float:
    """Realize an MLX scalar to a Python float."""
    return float(np.asarray(value))


def _metric_GD_MLX(front, context):
    """Registry GD on MLX: norm(min distances)/N, float32 device math."""
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
    nearest = _tiled_min_mlx(pop_m, opt_m)
    val = mx.sqrt(mx.sum(nearest * nearest)) / float(pop_m.shape[0])
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
    val = mx.mean(_tiled_min_mlx(opt_m, pop_m))
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
    val = mx.mean(_tiled_min_igdp_mlx(pop_m, opt_m, _cpu._igdp_p_from_context(context, default=2.0)))
    mx.eval(val)
    return _scalar(val)


def _metric_Spacing_MLX(front, context=None):
    """Registry Spacing on MLX: L1 nearest-neighbor std, ddof=1, float32 math."""
    pop = _cpu._get_front(front)
    if pop.size == 0:
        return float("nan")
    if pop.shape[0] <= 1:
        return 0.0
    pop_m = _to_mx(pop)
    val = _spacing_kernel_mlx(pop_m)
    mx.eval(val)
    return _scalar(val)


def _metric_DeltaP_MLX(front, context):
    """Registry DeltaP on MLX: max(registry GD, IGD), float32 device math."""
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
    nearest = _tiled_min_mlx(pop_m, opt_m)
    gd = mx.sqrt(mx.sum(nearest * nearest)) / float(pop_m.shape[0])
    igd = mx.mean(_tiled_min_mlx(opt_m, pop_m))
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
