# emopylab 2026
"""Size-adaptive, device-aware tensor kernels for the population-level primitives used by the ports.

Pairwise kernels (distances, angles, dominance, shift-based density) are O(N^2 M).  Small problems
(the usual N <= a few hundred) are fastest and bit-exact in float64 NumPy; once the number of pairwise
elements crosses ``EMOPYLAB_DEVICE_MIN_ELEMS`` (default 4e6) the work moves to the accelerator that the
process-wide tensor backend selected (PyTorch CUDA/ROCm/MPS or Apple MLX) in float32 with tiling, and the
result is returned as a NumPy array.  Any device failure falls back to the NumPy path.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__all__ = ["pdist2", "cosine_distance", "angle_matrix", "dominance_matrix", "sde_distance", "device_name"]

_MIN_ELEMS = int(float(os.environ.get("EMOPYLAB_DEVICE_MIN_ELEMS", 4e6)))
_TILE = 2 ** 25


def _backend() -> str:
    try:
        from core.tensor.backend import get_backend_type
        return str(get_backend_type())
    except Exception:  # noqa: BLE001
        return "numpy"


def device_name() -> str:
    return _backend()


def _torch_dev():
    from core.tensor.backend import _torch_device
    return _torch_device()


def _use_device(elems: int) -> bool:
    return elems >= _MIN_ELEMS and _backend() in ("torch", "mlx")


def _tiles(n: int, per_row: int):
    step = max(1, _TILE // max(1, per_row))
    return range(0, n, step), step


def _torch_call(fn):
    """Run ``fn(torch, device)`` and return NumPy; None when the device path fails."""
    try:
        import torch
        dev = _torch_dev()
        out = fn(torch, dev)
        return out.detach().cpu().numpy().astype(np.float64) if out.dtype.is_floating_point else out.detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        return None


def pdist2(A: Any, B: Any = None) -> np.ndarray:
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Bm = A if B is None else np.atleast_2d(np.asarray(B, dtype=float))
    n, m, d = len(A), len(Bm), A.shape[1]
    if _use_device(n * m * d):
        if _backend() == "torch":
            def f(torch, dev):
                a = torch.as_tensor(A, dtype=torch.float32, device=dev)
                b = torch.as_tensor(Bm, dtype=torch.float32, device=dev)
                return torch.cdist(a, b)
            r = _torch_call(f)
            if r is not None:
                return r
        else:
            try:
                import mlx.core as mx
                a, b = mx.array(A.astype(np.float32)), mx.array(Bm.astype(np.float32))
                step = max(1, _TILE // max(1, m * d))
                parts = []
                for s0 in range(0, n, step):  # direct differences: no catastrophic cancellation near 0
                    parts.append(mx.sqrt(mx.sum((a[s0:s0 + step, None, :] - b[None, :, :]) ** 2, axis=2)))
                r = mx.concatenate(parts, axis=0)
                mx.eval(r)
                return np.asarray(r, dtype=np.float64)
            except Exception:  # noqa: BLE001
                pass
    d2 = np.sum(A * A, 1)[:, None] + np.sum(Bm * Bm, 1)[None, :] - 2.0 * A @ Bm.T
    return np.sqrt(np.maximum(d2, 0.0))


def cosine_distance(A: Any, B: Any = None) -> np.ndarray:
    """``1 - cos`` between rows; zero vectors are treated as orthogonal (distance 1)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Bm = A if B is None else np.atleast_2d(np.asarray(B, dtype=float))
    if _use_device(len(A) * len(Bm) * A.shape[1]) and _backend() == "torch":
        def f(torch, dev):
            a = torch.as_tensor(A, dtype=torch.float32, device=dev)
            b = torch.as_tensor(Bm, dtype=torch.float32, device=dev)
            an, bn = a.norm(dim=1, keepdim=True), b.norm(dim=1, keepdim=True)
            cos = (a @ b.T) / (an * bn.T)
            return 1.0 - torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
        r = _torch_call(f)
        if r is not None:
            return r
    with np.errstate(all="ignore"):
        cos = (A @ Bm.T) / (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(Bm, axis=1)[None, :])
    return 1.0 - np.where(np.isfinite(cos), cos, 0.0)


def angle_matrix(A: Any, B: Any = None) -> np.ndarray:
    cos = 1.0 - cosine_distance(A, B)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def dominance_matrix(F: Any) -> np.ndarray:
    """``dom[i, j]`` True when solution i Pareto-dominates solution j (minimisation)."""
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    if _use_device(n * n * m) and _backend() == "torch":
        try:
            import torch
            dev = _torch_dev()
            f = torch.as_tensor(F, dtype=torch.float32, device=dev)
            rows = []
            for s in range(0, n, max(1, _TILE // max(1, n * m))):
                c = f[s:s + max(1, _TILE // max(1, n * m))]
                rows.append((torch.all(c[:, None, :] <= f[None], dim=-1) & torch.any(c[:, None, :] < f[None], dim=-1)).cpu())
            return torch.cat(rows).numpy()
        except Exception:  # noqa: BLE001
            pass
    out = np.empty((n, n), dtype=bool)
    step = max(1, _TILE // max(1, n * m))
    for s in range(0, n, step):
        c = F[s:s + step]
        out[s:s + step] = np.all(c[:, None, :] <= F[None], axis=2) & np.any(c[:, None, :] < F[None], axis=2)
    return out


def sde_distance(F: Any) -> np.ndarray:
    """Shift-based density distance ``D[i, j] = || f_i - max(f_j, f_i) ||`` (diagonal = inf)."""
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    if _use_device(n * n * m) and _backend() == "torch":
        def f(torch, dev):
            t = torch.as_tensor(F, dtype=torch.float32, device=dev)
            D = torch.empty((n, n), dtype=torch.float32, device=dev)
            step = max(1, _TILE // max(1, n * m))
            for s in range(0, n, step):
                c = t[s:s + step]
                D[s:s + step] = (c[:, None, :] - torch.maximum(t[None], c[:, None, :])).norm(dim=-1)
            D.fill_diagonal_(float("inf"))
            return D
        r = _torch_call(f)
        if r is not None:
            return r
    D = np.empty((n, n))
    step = max(1, _TILE // max(1, n * m))
    for s in range(0, n, step):
        c = F[s:s + step]
        D[s:s + step] = np.linalg.norm(c[:, None, :] - np.maximum(F[None], c[:, None, :]), axis=2)
    np.fill_diagonal(D, np.inf)
    return D
