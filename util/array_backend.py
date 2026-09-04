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


class _NumpyFacade:
    """Expose NumPy under the historical ``xp`` facade contract."""

    def __getattr__(self, item: str) -> Any:
        return getattr(_np, item)


xp = _NumpyFacade()

# Legacy compatibility names (historically CuPy-backed in some plugins).
cp = None
CUPY_AVAILABLE = False

# JAX-oriented aliases (project nomenclature).
jax_accel = None
JAX_ACCEL_AVAILABLE = False

# MLX-oriented aliases
mlx_accel = None
MLX_ACCEL_AVAILABLE = False


def to_numpy(value: Any) -> Any:
    """Convert arrays/scalars to NumPy arrays when possible."""
    if value is None:
        return None
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001
            pass
    return _np.asarray(value)


def to_device(value: Any, use_gpu: bool = False, dtype: Any = None) -> Any:
    """Return a NumPy array (CPU fallback)."""
    if value is None:
        return None
    if dtype is not None:
        return _np.asarray(value, dtype=dtype)
    return _np.asarray(value)


def get_array_module(_value: Any = None) -> Any:
    """Return the active array module (NumPy in this shim)."""
    return _np


def is_cupy_array(_value: Any) -> bool:
    """Compatibility helper: always false in the local CPU shim."""
    return False


def is_jax_array(_value: Any) -> bool:
    """Compatibility helper for JAX-oriented naming: always false here."""
    return False


def is_mlx_array(_value: Any) -> bool:
    """Compatibility helper for MLX-oriented naming: always false here."""
    return False


def backend_cdist(a: Any, b: Any, metric: str = "euclidean") -> _np.ndarray:
    """Distance matrix helper with SciPy fallback to pure NumPy for euclidean."""
    a_np = _np.asarray(to_numpy(a), dtype=float)
    b_np = _np.asarray(to_numpy(b), dtype=float)
    try:
        from scipy.spatial.distance import cdist

        return _np.asarray(cdist(a_np, b_np, metric=metric), dtype=float)
    except Exception:
        if str(metric).lower() != "euclidean":
            raise RuntimeError(f"Distance metric '{metric}' requires scipy.spatial.distance.cdist")
        diff = a_np[:, None, :] - b_np[None, :, :]
        return _np.linalg.norm(diff, axis=2)


def resolve_backend_config(
    *,
    use_gpu: bool = False,
    array_backend: str = "auto",
    gpu_dtype: str = "float32",
) -> dict[str, Any]:
    """
    Resolve backend settings for local algorithms.

    The physical shim is CPU-only. We still preserve the requested backend token
    so the caller can report intent (e.g., ``jax``) while running on CPU.
    """
    requested = str(array_backend).strip().lower() or "auto"
    if requested == "auto":
        requested = "jax" if bool(use_gpu) else "numpy"
    dtype = str(gpu_dtype).strip().lower()
    if dtype not in {"float32", "float64"}:
        dtype = "float32"
    return {
        "requested_backend": requested,
        "effective_backend": "numpy",
        "use_gpu": False,
        "gpu_dtype": dtype,
        "cupy_available": False,
        "jax_available": False,
        "mlx_available": False,
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
