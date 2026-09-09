"""PyTorch-accelerated community metrics (NVIDIA CUDA / AMD ROCm / Apple Silicon MPS).

Provides high-throughput, tensor-vectorized quality indicators for Evolutionary
Multi-Objective and Many-Objective Optimization. Designed for cross-platform
execution on both supercomputers (Santos Dumont LNCC, NVIDIA Tesla V100/A100)
and Apple Silicon (M1-M5 Metal Performance Shaders).
"""

from __future__ import annotations

from typing import Any
import numpy as np

from . import community_metrics as _cpu

try:
    import torch
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _get_torch_device() -> Any:
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_torch(values: Any, dev: Any = None) -> Any:
    if dev is None:
        dev = _get_torch_device()
    if isinstance(values, torch.Tensor):
        t = values.to(dev)
    else:
        arr = np.atleast_2d(np.asarray(values, dtype=np.float32))
        t = torch.as_tensor(arr, device=dev, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


_TILE_BUDGET = 2 ** 26


def _tiled_min_euclidean(aa: Any, bb: Any) -> Any:
    """Row-wise min Euclidean distance over query rows aa vs ref rows bb, tiled over bb."""
    n, k = int(aa.shape[0]), int(bb.shape[0])
    m = int(aa.shape[1])
    tile = max(1, min(k, _TILE_BUDGET // max(1, n * m)))
    best: Any | None = None
    for s in range(0, k, tile):
        chunk = bb[s:s + tile]
        cur = torch.min(torch.linalg.norm(aa.unsqueeze(1) - chunk.unsqueeze(0), dim=2), dim=1).values
        best = cur if best is None else torch.minimum(best, cur)
    return best


def _tiled_min_igdp_torch(pop_t: Any, opt_t: Any, p: float = 2.0) -> Any:
    """IGD+ row minima per OPT row, tiled over POP rows with context p-norm (matches CPU delta)."""
    n, k = int(pop_t.shape[0]), int(opt_t.shape[0])
    m = int(pop_t.shape[1])
    tile = max(1, min(n, _TILE_BUDGET // max(1, k * m)))
    pf = float(p)
    best: Any | None = None
    for s in range(0, n, tile):
        chunk = pop_t[s:s + tile]
        diff = torch.clamp(chunk.unsqueeze(0) - opt_t.unsqueeze(1), min=0.0)
        if abs(pf - 2.0) <= 1e-12:
            d = torch.linalg.norm(diff, dim=2)
        elif abs(pf - 1.0) <= 1e-12:
            d = torch.sum(diff, dim=2)
        else:
            d = torch.sum(torch.pow(diff, pf), dim=2) ** (1.0 / pf)
        cur = torch.min(d, dim=1).values
        best = cur if best is None else torch.minimum(best, cur)
    return best


def _pairwise_euclidean_torch(aa: Any, bb: Any) -> Any:
    diff = aa.unsqueeze(1) - bb.unsqueeze(0)
    return torch.linalg.norm(diff, dim=2)


def _metric_GD_Torch(front: Any, context: dict[str, Any]) -> float:
    """Registry GD on Torch: norm(min distances)/N, float32 device math."""
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None or pop.size == 0 or opt.size == 0:
        return float("nan")
    if not _HAS_TORCH:
        return _cpu._gd_value(pop, opt)
    if pop.shape[1] != opt.shape[1]:
        return float("nan")

    dev = _get_torch_device()
    pop_t = _to_torch(pop, dev)
    opt_t = _to_torch(opt, dev)

    nearest = _tiled_min_euclidean(pop_t, opt_t)
    gd = torch.linalg.vector_norm(nearest) / float(nearest.shape[0])
    return float(gd.item())


def _metric_IGD_Torch(front: Any, context: dict[str, Any]) -> float:
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None or pop.size == 0 or opt.size == 0:
        return float("nan")
    if not _HAS_TORCH:
        return _cpu._igd_value(pop, opt)
    if pop.shape[1] != opt.shape[1]:
        return float("nan")

    dev = _get_torch_device()
    pop_t = _to_torch(pop, dev)
    opt_t = _to_torch(opt, dev)

    igd = torch.mean(_tiled_min_euclidean(opt_t, pop_t))
    return float(igd.item())


def _metric_IGDp_Torch(front: Any, context: dict[str, Any]) -> float:
    """Registry IGD+ on Torch: directional max(pop-opt,0), context p norm."""
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None or pop.size == 0 or opt.size == 0:
        return float("nan")
    if not _HAS_TORCH:
        return _cpu._metric_IGDp(pop, context)
    if pop.shape[1] != opt.shape[1]:
        return float("nan")

    dev = _get_torch_device()
    pop_t = _to_torch(pop, dev)
    opt_t = _to_torch(opt, dev)

    p = _cpu._igdp_p_from_context(context, default=2.0)
    # Modified distance: max(pop - opt, 0) tiled over POP rows; result rows are OPT.
    igdp = torch.mean(_tiled_min_igdp_torch(pop_t, opt_t, p))
    return float(igdp.item())


def _metric_Spacing_Torch(front: Any, context: dict[str, Any] = None) -> float:
    """Registry Spacing on Torch: L1 nearest-neighbor std, ddof=1, float32 math."""
    pop = _cpu._get_front(front)
    if pop.size == 0:
        return float("nan")
    if pop.shape[0] <= 1:
        return 0.0
    if not _HAS_TORCH:
        return _cpu._metric_Spacing(pop, context)

    dev = _get_torch_device()
    pop_t = _to_torch(pop, dev)
    N = pop_t.shape[0]

    diff = torch.abs(pop_t.unsqueeze(1) - pop_t.unsqueeze(0))
    d = torch.sum(diff, dim=2)
    d = d + torch.eye(N, device=dev) * float("inf")
    min_d = torch.min(d, dim=1).values
    spacing = torch.std(min_d, unbiased=True)
    return float(spacing.item())


def _metric_DeltaP_Torch(front: Any, context: dict[str, Any]) -> float:
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None or pop.size == 0 or opt.size == 0:
        return float("nan")
    if pop.shape[1] != opt.shape[1]:
        return float("nan")

    dev = _get_torch_device()
    pop_t = _to_torch(pop, dev)
    opt_t = _to_torch(opt, dev)

    gd = torch.linalg.vector_norm(_tiled_min_euclidean(pop_t, opt_t)) / float(pop_t.shape[0])
    igd = torch.mean(_tiled_min_euclidean(opt_t, pop_t))

    val = torch.maximum(gd, igd)
    return float(val.item())


def _metric_HV_Torch(front: Any, context: dict[str, Any]) -> float:
    """HV via the unified HV_fast_MC implementation (PyTorch MPS/CUDA/CPU)."""
    from metrics.hv_fast_mc import HV_fast_MC
    pop_obj = _cpu._get_front(front)
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    return float(HV_fast_MC(pop_obj, optimum,
                            sample_num=int(context.get("hv_mc_samples", 10_000)),
                            context=context))


def evaluate_batch_igd_plus_torch(F_batch: Any, opt: Any) -> np.ndarray:
    """Evaluates IGD+ across a 3D batch [R, N, M] of runs in parallel on GPU."""
    if not _HAS_TORCH:
        F_np = np.asarray(F_batch, dtype=float)
        opt_np = np.asarray(opt, dtype=float)
        R = F_np.shape[0]
        results = np.empty(R, dtype=float)
        for r in range(R):
            diff = np.maximum(F_np[r][:, None, :] - opt_np[None, :, :], 0.0)
            d = np.linalg.norm(diff, axis=2)
            results[r] = np.mean(np.min(d, axis=0))
        return results

    dev = _get_torch_device()
    F_t = torch.as_tensor(F_batch, device=dev, dtype=torch.float32)
    opt_t = torch.as_tensor(opt, device=dev, dtype=torch.float32)

    # F_t: [R, N, M] -> [R, 1, N, M]
    # opt_t: [K, M] -> [1, K, 1, M]
    diff = torch.clamp(F_t.unsqueeze(1) - opt_t.unsqueeze(0).unsqueeze(2), min=0.0)
    d = torch.linalg.norm(diff, dim=3)  # [R, K, N]
    min_d = torch.min(d, dim=2).values   # [R, K]
    igdp = torch.mean(min_d, dim=1)      # [R]
    return igdp.detach().cpu().numpy()


METRICS = {
    "GD_Torch": _metric_GD_Torch,
    "HV_Torch": _metric_HV_Torch,
    "IGD_Torch": _metric_IGD_Torch,
    "IGDp_Torch": _metric_IGDp_Torch,
    "Spacing_Torch": _metric_Spacing_Torch,
    "DeltaP_Torch": _metric_DeltaP_Torch,
}
