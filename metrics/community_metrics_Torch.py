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


def _pairwise_euclidean_torch(aa: Any, bb: Any) -> Any:
    diff = aa.unsqueeze(1) - bb.unsqueeze(0)
    return torch.linalg.norm(diff, dim=2)


def _metric_GD_Torch(front: Any, context: dict[str, Any]) -> float:
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

    diff = pop_t.unsqueeze(1) - opt_t.unsqueeze(0)
    d = torch.linalg.norm(diff, dim=2)
    nearest = torch.min(d, dim=1).values
    gd = torch.sqrt(torch.sum(nearest * nearest)) / float(pop_t.shape[0])
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

    diff = opt_t.unsqueeze(1) - pop_t.unsqueeze(0)
    d = torch.linalg.norm(diff, dim=2)
    igd = torch.mean(torch.min(d, dim=1).values)
    return float(igd.item())


def _metric_IGDp_Torch(front: Any, context: dict[str, Any]) -> float:
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

    # Modified distance: max(pop - opt, 0)
    diff = torch.clamp(pop_t.unsqueeze(0) - opt_t.unsqueeze(1), min=0.0)
    d = torch.linalg.norm(diff, dim=2)
    igdp = torch.mean(torch.min(d, dim=1).values)
    return float(igdp.item())


def _metric_Spacing_Torch(front: Any, context: dict[str, Any] = None) -> float:
    pop = _cpu._get_front(front)
    if pop.size == 0 or pop.shape[0] < 2:
        return float("nan")
    if not _HAS_TORCH:
        return _cpu._metric_Spacing(pop, context)

    dev = _get_torch_device()
    pop_t = _to_torch(pop, dev)
    N = pop_t.shape[0]

    diff = torch.abs(pop_t.unsqueeze(1) - pop_t.unsqueeze(0))
    d = torch.sum(diff, dim=2)
    d = d + torch.eye(N, device=dev) * 1e9
    min_d = torch.min(d, dim=1).values
    d_bar = torch.mean(min_d)
    dev_sq = min_d - d_bar
    spacing = torch.sqrt(torch.sum(dev_sq * dev_sq) / float(N))
    return float(spacing.item())


def _metric_DeltaP_Torch(front: Any, context: dict[str, Any]) -> float:
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None or pop.size == 0 or opt.size == 0:
        return float("nan")
    if not _HAS_TORCH:
        return _cpu._deltap_value(pop, opt)
    if pop.shape[1] != opt.shape[1]:
        return float("nan")

    dev = _get_torch_device()
    pop_t = _to_torch(pop, dev)
    opt_t = _to_torch(opt, dev)

    d_pop_opt = _pairwise_euclidean_torch(pop_t, opt_t)
    gd = torch.sqrt(torch.sum(torch.min(d_pop_opt, dim=1).values ** 2)) / float(pop_t.shape[0])

    d_opt_pop = _pairwise_euclidean_torch(opt_t, pop_t)
    igd = torch.mean(torch.min(d_opt_pop, dim=1).values)

    val = torch.maximum(gd, igd)
    return float(val.item())


def _metric_HV_Torch(front: Any, context: dict[str, Any]) -> float:
    pop_obj = _cpu._get_front(front)
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _cpu._community_hv(pop_obj, optimum, context)


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
