from __future__ import annotations

import platform
import sys
from typing import Any


def detect_gpu_runtime() -> dict[str, Any]:
    info: dict[str, Any] = {
        "jax_ok": False,
        "jax_version": None,
        "cuda_ok": False,
        "cuda_device_name": None,
        "device_count": 0,
        "all_devices": [],
        "error": None,
    }
    try:
        import jax
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"JAX not available: {exc}"
        return info

    info["jax_ok"] = True
    info["jax_version"] = str(getattr(jax, "__version__", "unknown"))
    try:
        devices = list(jax.devices())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"Could not query JAX devices: {exc}"
        return info

    info["device_count"] = len(devices)
    info["all_devices"] = [f"{getattr(d, 'platform', '?')}:{getattr(d, 'device_kind', d)}" for d in devices]

    gpu_devices = [d for d in devices if str(getattr(d, "platform", "")).lower() in {"gpu", "cuda", "rocm"}]
    if gpu_devices:
        info["cuda_ok"] = True
        info["cuda_device_name"] = str(getattr(gpu_devices[0], "device_kind", gpu_devices[0]))

    return info


def build_gpu_status_text(info: dict[str, Any]) -> str:
    if info.get("cuda_ok"):
        return f"JAX GPU available: {info.get('cuda_device_name', 'GPU device')}"
    if info.get("jax_ok"):
        return "JAX available, but no GPU device detected. CPU execution will be used."
    if info.get("error"):
        return str(info.get("error"))
    return "JAX runtime unavailable. CPU execution will be used."


def apple_silicon_available() -> bool:
    """Return True when running on Apple Silicon (macOS, arm64)."""
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def detect_mlx_runtime() -> dict[str, Any]:
    """Detect the MLX runtime without importing it eagerly.

    MLX is Apple's array framework for Apple Silicon. It is an optional
    acceleration backend; this probe never raises and reports availability,
    version, and the default device when present.
    """
    info: dict[str, Any] = {
        "mlx_ok": False,
        "mlx_version": None,
        "device": None,
        "apple_silicon": apple_silicon_available(),
        "error": None,
    }
    try:
        import mlx.core as mx
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"MLX not available: {exc}"
        return info

    info["mlx_ok"] = True
    try:
        import mlx

        info["mlx_version"] = str(getattr(mlx, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        info["mlx_version"] = "unknown"

    try:
        info["device"] = str(mx.default_device())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"Could not query MLX device: {exc}"

    return info

def detect_torch_runtime() -> dict[str, Any]:
    """Detect PyTorch runtime across NVIDIA CUDA, AMD ROCm, and Apple Silicon MPS."""
    info: dict[str, Any] = {
        "torch_ok": False,
        "torch_version": None,
        "cuda_ok": False,
        "rocm_ok": False,
        "mps_ok": False,
        "device_name": None,
        "device_count": 0,
        "all_devices": [],
        "error": None,
    }
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"PyTorch not available: {exc}"
        return info

    info["torch_ok"] = True
    info["torch_version"] = str(getattr(torch, "__version__", "unknown"))

    if torch.cuda.is_available():
        is_hip = getattr(torch.version, "hip", None) is not None
        info["cuda_ok"] = not is_hip
        info["rocm_ok"] = is_hip
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["all_devices"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_ok"] = True
        info["device_count"] = 1
        info["device_name"] = "Apple Silicon MPS (Metal)"
        info["all_devices"] = ["Apple MPS"]
    else:
        info["device_name"] = "CPU"
        info["device_count"] = 1
        info["all_devices"] = ["CPU"]

    return info


def build_torch_status_text(info: dict[str, Any]) -> str:
    if info.get("cuda_ok"):
        return f"PyTorch CUDA available: {info.get('device_name', 'NVIDIA GPU')} ({info.get('device_count', 1)} devices)"
    if info.get("rocm_ok"):
        return f"PyTorch ROCm available: {info.get('device_name', 'AMD GPU')} ({info.get('device_count', 1)} devices)"
    if info.get("mps_ok"):
        return f"PyTorch MPS available: {info.get('device_name', 'Apple Silicon Metal')}"
    if info.get("torch_ok"):
        return "PyTorch available (CPU execution)."
    if info.get("error"):
        return str(info.get("error"))
    return "PyTorch unavailable. CPU execution will be used."


def build_backend_status_text(info: dict[str, Any], backend: str = "jax") -> str:
    """Backend-agnostic status text dispatcher.

    Routes to the MLX/CPU messages, or to the existing JAX wording for the
    ``jax``/``gpu`` tokens so legacy callers stay unchanged.
    """
    token = str(backend or "").strip().lower()
    if token in ("torch", "pytorch", "mps"):
        return build_torch_status_text(info)
    if token == "mlx":
        if info.get("mlx_ok") and info.get("device"):
            return f"MLX available: {info.get('device')}"
        if info.get("mlx_ok"):
            return "MLX available (device unknown)."
        if info.get("error"):
            return str(info.get("error"))
        return "MLX runtime unavailable. CPU execution will be used."
    if token == "cpu":
        return "CPU execution (NumPy)."
    return build_gpu_status_text(info)
