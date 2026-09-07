# Made by EmoPyLab 2026.
"""
Local array-backend compatibility shim for legacy/community algorithms.

This module provides the subset of the historical ``util.array_backend``
interface expected by several local algorithms. The current project runtime
uses JAX as the accelerated backend at the application level, but many local
algorithms still import legacy CuPy-like symbols (``cp``, ``CUPY_AVAILABLE``)
for compatibility.

This physical module exists so algorithms can be imported without requiring
``EmoPyLab.py`` to inject a dynamic shim into ``sys.modules`` first.
"""

from __future__ import annotations

from typing import Any

import numpy as _np


class _DynamicArrayFacade:
    """Expose the active accelerator array module dynamically under historical xp contract."""

    def __getattr__(self, item: str) -> Any:
        from core.tensor.backend import get_array_module
        return getattr(get_array_module(), item)


xp = _DynamicArrayFacade()

# Dynamic compatibility properties
def _check_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False

def _check_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False

CUPY_AVAILABLE = _check_cupy()
JAX_ACCEL_AVAILABLE = False
MLX_ACCEL_AVAILABLE = _check_mlx()
cp = None
if CUPY_AVAILABLE:
    try:
        import cupy as cp
    except Exception:
        pass

def to_numpy(value: Any) -> Any:
    """Convert arrays/scalars to NumPy arrays when possible."""
    if value is None:
        return None
    from core.tensor.backend import to_numpy as _core_to_numpy
    return _core_to_numpy(value)


def to_device(value: Any, use_gpu: bool = False, dtype: Any = None) -> Any:
    """Transfer array to active acceleration device if requested, else NumPy."""
    if value is None:
        return None
    if use_gpu:
        from core.tensor.backend import to_device as _core_to_device
        return _core_to_device(value, dtype=dtype)
    if dtype is not None:
        return _np.asarray(to_numpy(value), dtype=dtype)
    return _np.asarray(to_numpy(value))


def get_array_module(_value: Any = None) -> Any:
    """Return the active array module (MLX, PyTorch, or NumPy)."""
    from core.tensor.backend import get_array_module as _core_get_array_module
    return _core_get_array_module()


def is_cupy_array(value: Any) -> bool:
    if value is None:
        return False
    return "cupy" in type(value).__module__


def is_jax_array(value: Any) -> bool:
    if value is None:
        return False
    return "jax" in type(value).__module__


def is_mlx_array(value: Any) -> bool:
    if value is None:
        return False
    return "mlx" in type(value).__module__


def is_torch_array(value: Any) -> bool:
    if value is None:
        return False
    return "torch" in type(value).__module__

def backend_cdist(a: Any, b: Any, metric: str = "euclidean") -> Any:
    """Distance matrix helper accelerated on active GPU/NPU when available."""
    from core.tensor.backend import get_backend_type, get_array_module
    btype = get_backend_type()
    if str(metric).lower() == "euclidean":
        if btype == "mlx":
            import mlx.core as mx
            aa = mx.array(a) if not is_mlx_array(a) else a
            bb = mx.array(b) if not is_mlx_array(b) else b
            diff = mx.expand_dims(aa, 1) - mx.expand_dims(bb, 0)
            return mx.sqrt(mx.sum(diff * diff, axis=2))
        elif btype == "torch":
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
            aa = torch.as_tensor(a, device=dev) if not is_torch_array(a) else a.to(dev)
            bb = torch.as_tensor(b, device=dev) if not is_torch_array(b) else b.to(dev)
            diff = aa.unsqueeze(1) - bb.unsqueeze(0)
            return torch.linalg.norm(diff, dim=2)

    a_np = _np.asarray(to_numpy(a), dtype=float)
    b_np = _np.asarray(to_numpy(b), dtype=float)
    try:
        from scipy.spatial.distance import cdist
        return _np.asarray(cdist(a_np, b_np, metric=metric), dtype=float)
    except Exception:
        diff = a_np[:, None, :] - b_np[None, :, :]
        return _np.linalg.norm(diff, axis=2)


def resolve_backend_config(
    *,
    use_gpu: bool = False,
    array_backend: str = "auto",
    gpu_dtype: str = "float32",
) -> dict[str, Any]:
    """Resolve backend settings using core tensor backend detection."""
    from core.tensor.backend import init_tensor_backend, get_backend_type, _DEVICE_INFO
    requested = str(array_backend).strip().lower() or "auto"
    if requested == "auto":
        info = init_tensor_backend(prefer=None)
    else:
        info = init_tensor_backend(prefer=requested if requested in ("torch", "mlx", "cupy", "jax", "numpy") else None)

    is_accel = bool(info.get("accelerated", False))
    eff = info.get("backend", "numpy")
    return {
        "requested_backend": requested,
        "effective_backend": eff,
        "use_gpu": is_accel if use_gpu or requested != "numpy" else False,
        "gpu_dtype": gpu_dtype,
        "cupy_available": CUPY_AVAILABLE,
        "jax_available": False,
        "mlx_available": MLX_ACCEL_AVAILABLE,
        "torch_available": True,
    }

def get_cupy_device_name(_device_id: int = 0) -> str | None:
    """Legacy compatibility helper (no accelerator in this shim)."""
    return None


def get_jax_device_name(_device_id: int = 0) -> str | None:
    """JAX-oriented compatibility alias (no accelerator in this shim)."""
    return None


__all__ = [
    "xp",
    "cp",
    "CUPY_AVAILABLE",
    "jax_accel",
    "JAX_ACCEL_AVAILABLE",
    "mlx_accel",
    "MLX_ACCEL_AVAILABLE",
    "to_numpy",
    "to_device",
    "get_array_module",
    "is_cupy_array",
    "is_jax_array",
    "is_mlx_array",
    "backend_cdist",
    "resolve_backend_config",
    "get_cupy_device_name",
    "get_jax_device_name",
]
