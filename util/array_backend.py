# Made by EmoPyLab 2026.
"""Local array-backend compatibility facade over the core tensor backend.

CuPy/Torch/MLX/JAX remain optional; every availability probe runs lazily so
this module imports without any accelerator installed.
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

def _check_cupy() -> bool:
    """Probe CuPy without importing it eagerly at module scope."""
    try:
        import importlib.util
        return importlib.util.find_spec("cupy") is not None
    except Exception:
        return False


def _available_via_registry(probe: str) -> bool:
    try:
        from core.registry import backends as _backends
        return bool(getattr(_backends, probe)())
    except Exception:
        return False


def _check_mlx() -> bool:
    return _available_via_registry("mlx_available")


def _check_jax_accel() -> bool:
    try:
        import jax
        return str(jax.default_backend()).lower() in {"gpu", "cuda", "rocm", "metal", "tpu"}
    except Exception:
        return False


def _cupy_module() -> Any:
    try:
        import cupy as _cupy
        return _cupy
    except Exception:
        return None


CUPY_AVAILABLE = _check_cupy()
JAX_ACCEL_AVAILABLE = _check_jax_accel()
MLX_ACCEL_AVAILABLE = _check_mlx()
cp = _cupy_module()
def to_numpy(value: Any) -> Any:
    """Convert arrays/scalars to NumPy arrays when possible."""
    if value is None:
        return None
    from core.tensor.backend import to_numpy as _core_to_numpy
    return _core_to_numpy(value)


def to_device(value: Any, use_gpu: bool = False, dtype: Any = None) -> Any:
    """Transfer to the active device when requested, else NumPy."""
    if value is None:
        return None
    if use_gpu:
        from core.tensor.backend import to_device as _core_to_device
        return _core_to_device(value, dtype=dtype)
    if dtype is not None:
        return _np.asarray(to_numpy(value), dtype=dtype)
    return _np.asarray(to_numpy(value))


def get_array_module(_value: Any = None) -> Any:
    """Return the active array module."""
    from core.tensor.backend import get_array_module as _core_get_array_module
    return _core_get_array_module()


def is_cupy_array(value: Any) -> bool:
    if value is None:
        return False
    return "cupy" in type(value).__module__


def is_jax_array(value: Any) -> bool:
    if value is None:
        return False
def backend_cdist(a: Any, b: Any, metric: str = "euclidean") -> Any:
    """Pairwise distances on the active backend without changing precision."""
    from core.tensor.backend import get_array_module, get_backend_type, to_device, to_numpy
    btype = get_backend_type()
    if str(metric).lower() == "euclidean":
        if btype == "mlx":
            aa = to_device(_np.asarray(to_numpy(a), dtype=np.float32))
            bb = to_device(_np.asarray(to_numpy(b), dtype=np.float32))
            import mlx.core as mx
            result = mx.sqrt(mx.sum((mx.expand_dims(aa, 1) - mx.expand_dims(bb, 0)) ** 2, axis=2))
            mx.eval(result)
            return result
        if btype == "torch":
            aa = to_device(_np.asarray(to_numpy(a), dtype=np.float32))
            bb = to_device(_np.asarray(to_numpy(b), dtype=np.float32))
            return get_array_module().linalg.norm(aa.unsqueeze(1) - bb.unsqueeze(0), dim=2)
        if btype == "cupy":
            aa = to_device(_np.asarray(to_numpy(a), dtype=np.float32))
            bb = to_device(_np.asarray(to_numpy(b), dtype=np.float32))
            return get_array_module().linalg.norm(aa[:, None, :] - bb[None, :, :], axis=2)

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
    from core.tensor.backend import init_tensor_backend
    from core.registry.backends import jax_available, mlx_available, torch_available
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
        "jax_available": jax_available(),
        "mlx_available": mlx_available(),
        "torch_available": torch_available(),
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
