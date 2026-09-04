"""EmoPyLab Multi-Backend Tensor Layer (GPU / NPU / Vectorized CPU).

Supports JAX (CUDA/ROCm/Apple Metal/CPU XLA), Apple MLX (Neural Engine),
and optimized NumPy C-SIMD fallback with zero-overhead dispatch.
"""

from __future__ import annotations
import os
import sys
from types import ModuleType
from typing import Any, Callable, Literal, Sequence

import numpy as np

BackendType = Literal["torch", "cupy", "jax", "mlx", "numpy"]

_ACTIVE_BACKEND: BackendType = "numpy"
_ARRAY_MODULE: Any = np
_JIT_COMPILER: Callable[[Callable], Callable] = lambda f: f
_DEVICE_INFO: dict[str, Any] = {}


def get_backend_type() -> BackendType:
    return _ACTIVE_BACKEND


def get_array_module() -> Any:
    global _ARRAY_MODULE
    if _ARRAY_MODULE is None:
        init_tensor_backend()
    return _ARRAY_MODULE


def jit(fn: Callable) -> Callable:
    """Decorator to apply JIT compilation if the active backend supports it."""
    global _JIT_COMPILER
    return _JIT_COMPILER(fn)


def init_tensor_backend(prefer: BackendType | None = None) -> dict[str, Any]:
    """Initialize and return the tensor backend runtime info.

    Auto-detects available accelerators with transparent CPU fallback:
      1. NVIDIA CUDA / PyTorch (if torch is installed with CUDA available)
      2. CuPy (if cupy is installed with CUDA available)
      3. JAX (CUDA / ROCm / TPU / Apple Metal MPS)
      4. Apple MLX (Apple Silicon GPU / Neural Engine)
      5. Vectorized NumPy (C-SIMD / OpenBLAS / MKL CPU fallback)
    """
    global _ACTIVE_BACKEND, _ARRAY_MODULE, _JIT_COMPILER, _DEVICE_INFO

    is_apple_arm = sys.platform == "darwin" and os.uname().machine in ("arm64", "aarch64")

    # 1. Try PyTorch CUDA / MPS
    if prefer in (None, "torch"):
        try:
            import torch
            if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and prefer == "torch"):
                device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Apple Silicon MPS"
                _ACTIVE_BACKEND = "torch"
                _ARRAY_MODULE = torch
                _JIT_COMPILER = getattr(torch, "compile", lambda f: f)
                _DEVICE_INFO = {
                    "backend": "torch",
                    "platform": "cuda" if torch.cuda.is_available() else "mps",
                    "devices": [torch.cuda.get_device_name(i) for i in range(device_count)] if torch.cuda.is_available() else ["Apple MPS"],
                    "device_count": device_count,
                    "is_gpu": True,
                    "accelerated": True,
                    "gpu_vendor": "NVIDIA" if torch.cuda.is_available() else "Apple",
                    "gpu_name": gpu_name,
                }
                return _DEVICE_INFO
        except Exception:
            pass

    # 2. Try CuPy (NVIDIA CUDA)
    if prefer in (None, "cupy"):
        try:
            import cupy as cp
            dev_count = cp.cuda.runtime.getDeviceCount()
            if dev_count > 0:
                _ACTIVE_BACKEND = "cupy"
                _ARRAY_MODULE = cp
                _JIT_COMPILER = lambda f: f
                _DEVICE_INFO = {
                    "backend": "cupy",
                    "platform": "cuda",
                    "devices": [f"CUDA Device {i}" for i in range(dev_count)],
                    "device_count": dev_count,
                    "is_gpu": True,
                    "accelerated": True,
                    "gpu_vendor": "NVIDIA",
                    "gpu_name": f"NVIDIA CUDA Device ({dev_count} GPUs)",
                }
                return _DEVICE_INFO
        except Exception:
            pass

    # 3. On Apple Silicon, prioritize native MLX (Unified Memory GPU & Neural Engine)
    if is_apple_arm and prefer in (None, "mlx"):
        try:
            import mlx.core as mx
            _ACTIVE_BACKEND = "mlx"
            _ARRAY_MODULE = mx
            _JIT_COMPILER = mx.compile
            _DEVICE_INFO = {
                "backend": "mlx",
                "platform": "apple_silicon",
                "devices": ["Apple Silicon GPU / Neural Engine (Device(gpu, 0))"],
                "device_count": 1,
                "is_gpu": True,
                "accelerated": True,
                "gpu_vendor": "Apple",
                "gpu_name": "Apple Silicon GPU / Neural Engine (MLX)",
            }
            return _DEVICE_INFO
        except Exception:
            pass

    # 4. Try JAX (CUDA / ROCm / Apple Metal MPS / CPU XLA)
    if prefer in (None, "jax"):
        try:
            import jax
            import jax.numpy as jnp

            devices = jax.devices()
            platform = jax.default_backend()
            
            # Detect GPU vendor from device info
            gpu_devices = [d for d in devices if "gpu" in str(getattr(d, "platform", "")).lower()]
            gpu_vendor = None
            gpu_name = None
            if gpu_devices:
                d = gpu_devices[0]
                device_str = str(d)
                kind = str(getattr(d, "device_kind", ""))
                if "cuda" in platform or "cuda" in device_str.lower() or "nvidia" in device_str.lower():
                    gpu_vendor = "NVIDIA"
                    gpu_name = f"NVIDIA GPU ({kind or device_str})"
                elif "rocm" in platform or "rocm" in device_str.lower() or "amd" in device_str.lower():
                    gpu_vendor = "AMD"
                    gpu_name = f"AMD GPU ({kind or device_str})"
                elif "metal" in platform or "apple" in device_str.lower():
                    gpu_vendor = "Apple"
                    gpu_name = f"Apple GPU / Neural Engine ({kind or device_str})"
                else:
                    gpu_name = f"GPU ({kind or device_str})"
            else:
                gpu_name = None
            
            is_gpu = platform in ("gpu", "cuda", "rocm", "metal", "tpu")
            
            # If JAX is running on pure CPU and MLX is available on Apple Silicon, prefer MLX
            if not is_gpu and is_apple_arm and prefer is None:
                try:
                    import mlx.core as mx
                    _ACTIVE_BACKEND = "mlx"
                    _ARRAY_MODULE = mx
                    _JIT_COMPILER = mx.compile
                    _DEVICE_INFO = {
                        "backend": "mlx",
                        "platform": "apple_silicon",
                        "devices": ["Apple Silicon GPU / Neural Engine (Device(gpu, 0))"],
                        "device_count": 1,
                        "is_gpu": True,
                        "accelerated": True,
                        "gpu_vendor": "Apple",
                        "gpu_name": "Apple Silicon GPU / Neural Engine (MLX)",
                    }
                    return _DEVICE_INFO
                except Exception:
                    pass

            _ACTIVE_BACKEND = "jax"
            _ARRAY_MODULE = jnp
            _JIT_COMPILER = jax.jit
            _DEVICE_INFO = {
                "backend": "jax",
                "platform": platform,
                "devices": [str(d) for d in devices],
                "device_count": len(devices),
                "is_gpu": is_gpu,
                "accelerated": is_gpu,
                "gpu_vendor": gpu_vendor,
                "gpu_name": gpu_name,
            }
            return _DEVICE_INFO
        except Exception:
            pass

    # 5. Fallback to NumPy C-SIMD
    _ACTIVE_BACKEND = "numpy"
    _ARRAY_MODULE = np
    _JIT_COMPILER = lambda f: f
    _DEVICE_INFO = {
        "backend": "numpy",
        "platform": "cpu",
        "devices": ["CPU"],
        "device_count": os.cpu_count() or 1,
        "is_gpu": False,
        "accelerated": False,
    }
    return _DEVICE_INFO


def to_device(array: Any, dtype: Any = None) -> Any:
    """Transfers host data into a contiguous device tensor, preserving bool/int types."""
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        if isinstance(array, torch.Tensor):
            return array.to(dev)
        t = torch.as_tensor(array, device=dev)
        if dtype is not None:
            return t.to(dtype=dtype)
        return t
    elif _ACTIVE_BACKEND == "cupy":
        import cupy as cp
        return cp.asarray(array, dtype=dtype)
    elif _ACTIVE_BACKEND == "jax":
        if dtype is not None:
            return xp.asarray(array, dtype=dtype)
        return xp.asarray(array)
    elif _ACTIVE_BACKEND == "mlx":
        return xp.array(array)
    if dtype is not None:
        return np.ascontiguousarray(array, dtype=dtype)
    return np.ascontiguousarray(array)


def to_numpy(tensor: Any) -> np.ndarray:
    """Converts any backend tensor back into a standard CPU NumPy array."""
    if tensor is None:
        return np.empty((0, 0), dtype=np.float32)
    if isinstance(tensor, np.ndarray):
        return tensor
    if _ACTIVE_BACKEND == "torch":
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)
    if _ACTIVE_BACKEND == "cupy":
        import cupy as cp
        if isinstance(tensor, cp.ndarray):
            return cp.asnumpy(tensor)
        return np.asarray(tensor)
    if _ACTIVE_BACKEND == "jax":
        return np.asarray(tensor)
    if _ACTIVE_BACKEND == "mlx":
        return np.array(tensor)
    return np.asarray(tensor)


def zeros(shape: Sequence[int], dtype: Any = np.float32) -> Any:
    """Creates a zero tensor on the active device."""
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        return torch.zeros(shape, device=dev)
    if _ACTIVE_BACKEND == "mlx":
        import mlx.core as mx
        mlx_dtype = mx.float32 if dtype in (np.float32, float, "float32") else (mx.int32 if dtype in (np.int32, int, "int32") else mx.float32)
        return xp.zeros(shape, dtype=mlx_dtype)
    return xp.zeros(shape, dtype=dtype)


def ones(shape: Sequence[int], dtype: Any = np.float32) -> Any:
    """Creates an ones tensor on the active device."""
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        return torch.ones(shape, device=dev)
    if _ACTIVE_BACKEND == "mlx":
        import mlx.core as mx
        mlx_dtype = mx.float32 if dtype in (np.float32, float, "float32") else (mx.int32 if dtype in (np.int32, int, "int32") else mx.float32)
        return xp.ones(shape, dtype=mlx_dtype)
    return xp.ones(shape, dtype=dtype)


def clip_bounds(x: Any, xl: Any, xu: Any) -> Any:
    """Element-wise box constraint clamping on active device."""
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        import torch
        x_t = to_device(x)
        xl_t = to_device(xl)
        xu_t = to_device(xu)
        return torch.clamp(x_t, xl_t, xu_t)
    if _ACTIVE_BACKEND == "mlx":
        x_dev = to_device(x)
        xl_dev = to_device(xl)
        xu_dev = to_device(xu)
        return xp.clip(x_dev, xl_dev, xu_dev)
    return xp.clip(x, xl, xu)


def hstack(tensors: Sequence[Any]) -> Any:
    """Horizontally stacks tensors along the last dimension on active device."""
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        import torch
        dev_tensors = [to_device(t) for t in tensors]
        return torch.cat(dev_tensors, dim=-1)
    if _ACTIVE_BACKEND == "mlx":
        import mlx.core as mx
        dev_tensors = [to_device(t) for t in tensors]
        return mx.concatenate(dev_tensors, axis=-1)
    return xp.hstack(tensors)


def vstack(tensors: Sequence[Any]) -> Any:
    """Vertically stacks tensors along axis 0 on active device."""
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        import torch
        dev_tensors = [to_device(t) for t in tensors]
        return torch.cat(dev_tensors, dim=0)
    if _ACTIVE_BACKEND == "mlx":
        import mlx.core as mx
        dev_tensors = [to_device(t) for t in tensors]
        return mx.concatenate(dev_tensors, axis=0)
    return xp.vstack(tensors)


def index_tensor(tensor: Any, indices: Any) -> Any:
    """Indexes into a tensor along axis 0 across PyTorch, CuPy, MLX, JAX, and NumPy."""
    if tensor is None:
        return None
    if _ACTIVE_BACKEND == "mlx":
        import mlx.core as mx
        idx = mx.array(np.asarray(to_numpy(indices), dtype=np.int32).reshape(-1))
        return tensor[idx]
    if _ACTIVE_BACKEND == "torch":
        import torch
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(to_numpy(indices), dtype=torch.long, device=tensor.device if hasattr(tensor, "device") else None)
        return tensor[indices]
    idx = np.asarray(to_numpy(indices), dtype=np.int64).reshape(-1)
    return tensor[idx]


def array_copy(x: Any) -> Any:
    """Copies an array or tensor preserving backend type."""
    if _ACTIVE_BACKEND == "torch":
        import torch
        if isinstance(x, torch.Tensor):
            return x.clone()
        return torch.tensor(x).clone()
    if _ACTIVE_BACKEND == "mlx":
        import mlx.core as mx
        if isinstance(x, mx.array):
            return mx.array(x)
    xp = get_array_module()
    return xp.copy(x) if hasattr(xp, "copy") else np.copy(x)
