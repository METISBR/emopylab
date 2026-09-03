"""Backend descriptor registry (Phase 1 of MLX support).

Generalizes the historically binary CPU/JAX choice into a small registry of
named backends. Each backend carries the metadata needed to resolve variant
artifacts (the ``_JAX`` / ``_MLX`` suffix convention), report availability, and
probe its runtime. Availability checks import accelerators lazily and never
raise, so importing this module is always safe and CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.execution.backend_runtime import (
    apple_silicon_available,
    detect_gpu_runtime,
    detect_mlx_runtime,
)


def cpu_available() -> bool:
    return True


def jax_available() -> bool:
    try:
        import jax  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def detect_cpu_runtime() -> dict[str, Any]:
    return {"cpu_ok": True, "error": None}


@dataclass(frozen=True)
class BackendDescriptor:
    """Static metadata describing one acceleration backend."""

    token: str                       # canonical token: "cpu" | "jax" | "mlx"
    label: str                       # human-facing label shown in the UI
    suffix: str                      # variant suffix: "" | "JAX" | "MLX"
    default_dtype: str               # preferred floating dtype
    is_available: Callable[[], bool]  # lazy availability probe
    detect_runtime: Callable[[], dict[str, Any]]


BACKENDS: dict[str, BackendDescriptor] = {
    "cpu": BackendDescriptor(
        "cpu", "CPU (NumPy)", "", "float64", cpu_available, detect_cpu_runtime
    ),
    "jax": BackendDescriptor(
        "jax", "JAX", "JAX", "float32", jax_available, detect_gpu_runtime
    ),
    "mlx": BackendDescriptor(
        "mlx", "MLX (Apple Silicon)", "MLX", "float32", mlx_available, detect_mlx_runtime
    ),
}

# UI/config tokens that map onto a canonical backend.
_BACKEND_ALIASES = {
    "": "cpu",
    "cpu": "cpu",
    "numpy": "cpu",
    "np": "cpu",
    "auto": "cpu",
    "gpu": "jax",
    "jax": "jax",
    "cuda": "jax",
    "rocm": "jax",
    "mlx": "mlx",
    "metal": "mlx",
    "apple": "mlx",
}


def normalize_backend(value: Any) -> str:
    """Map any UI/config token to a canonical backend token (defaults to cpu)."""
    token = str(value or "").strip().lower()
    return _BACKEND_ALIASES.get(token, "cpu")


def backend_descriptor(value: Any) -> BackendDescriptor:
    return BACKENDS[normalize_backend(value)]


def backend_suffix(value: Any) -> str:
    """Variant suffix for a backend token (e.g., 'JAX', 'MLX', or '')."""
    return backend_descriptor(value).suffix


def backend_label(value: Any) -> str:
    return backend_descriptor(value).label


def all_variant_suffixes() -> tuple[str, ...]:
    """Non-empty suffixes of every accelerated backend (used for CPU filtering)."""
    return tuple(d.suffix for d in BACKENDS.values() if d.suffix)


def available_backends(*, require_apple_silicon_for_mlx: bool = True) -> list[str]:
    """Tokens of currently usable backends, honoring Apple Silicon gating for MLX."""
    tokens = ["cpu"]
    if jax_available():
        tokens.append("jax")
    if mlx_available() and (apple_silicon_available() or not require_apple_silicon_for_mlx):
        tokens.append("mlx")
    return tokens


__all__ = [
    "BackendDescriptor",
    "BACKENDS",
    "cpu_available",
    "jax_available",
    "mlx_available",
    "detect_cpu_runtime",
    "normalize_backend",
    "backend_descriptor",
    "backend_suffix",
    "backend_label",
    "all_variant_suffixes",
    "available_backends",
]
