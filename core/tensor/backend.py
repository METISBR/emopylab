"""EmoPyLab Multi-Backend Tensor Layer (GPU / Vectorized CPU).

Supports JAX (CUDA/ROCm/Apple Metal/CPU XLA), Apple MLX (GPU/Metal),
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
      1. Apple MLX on Apple Silicon (Unified Memory GPU/Metal)
      2. NVIDIA CUDA / AMD ROCm / Apple MPS via PyTorch
      3. NVIDIA CUDA via CuPy
      4. JAX (CUDA / ROCm / TPU / Apple Metal MPS)
      5. Vectorized NumPy (C-SIMD / OpenBLAS / Accelerate / MKL CPU fallback)
    """
    global _ACTIVE_BACKEND, _ARRAY_MODULE, _JIT_COMPILER, _DEVICE_INFO

    is_apple_arm = sys.platform == "darwin" and os.uname().machine in ("arm64", "aarch64")

    # Priority 1 on Apple Silicon: Native MLX (Unified Memory GPU/Metal; no direct ANE API)
    if is_apple_arm and prefer in (None, "mlx"):
        try:
            import mlx.core as mx
            _ACTIVE_BACKEND = "mlx"
            _ARRAY_MODULE = mx
            _JIT_COMPILER = mx.compile
            _DEVICE_INFO = {
                "backend": "mlx",
                "platform": "apple_silicon",
                "devices": ["Apple Silicon GPU/Metal (Device(gpu, 0))"],
                "device_count": 1,
                "is_gpu": True,
                "accelerated": True,
                "gpu_vendor": "Apple",
                "gpu_name": "Apple Silicon GPU/Metal (MLX)",
            }
            return _DEVICE_INFO
        except Exception:
            pass

    # Priority 2: PyTorch (NVIDIA CUDA, AMD ROCm/HIP, or Apple Silicon MPS)
    if prefer in (None, "torch"):
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if has_cuda or has_mps or prefer == "torch":
                device_count = torch.cuda.device_count() if has_cuda else 1
                if has_cuda:
                    is_hip = getattr(torch.version, "hip", None) is not None
                    gpu_vendor = "AMD" if is_hip else "NVIDIA"
                    gpu_name = torch.cuda.get_device_name(0)
                    platform_name = "rocm" if is_hip else "cuda"
                    dev_list = [torch.cuda.get_device_name(i) for i in range(device_count)]
                elif has_mps:
                    gpu_vendor = "Apple"
                    gpu_name = "Apple Silicon MPS (Metal Performance Shaders)"
                    platform_name = "mps"
                    dev_list = ["Apple MPS"]
                else:
                    gpu_vendor = "CPU"
                    gpu_name = "PyTorch CPU Engine"
                    platform_name = "cpu"
                    dev_list = ["CPU"]

                _ACTIVE_BACKEND = "torch"
                _ARRAY_MODULE = torch
                _JIT_COMPILER = getattr(torch, "compile", lambda f: f)
                _DEVICE_INFO = {
                    "backend": "torch",
                    "platform": platform_name,
                    "devices": dev_list,
                    "device_count": device_count,
                    "is_gpu": has_cuda or has_mps,
                    "accelerated": has_cuda or has_mps,
                    "gpu_vendor": gpu_vendor,
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
                    gpu_name = f"Apple GPU/Metal ({kind or device_str})"
                else:
                    gpu_name = f"GPU ({kind or device_str})"
            else:
                gpu_name = None
            
            is_gpu = platform in ("gpu", "cuda", "rocm", "metal", "tpu")
            # CPU-only JAX is not an accelerator: fall through to NumPy CPU
            # fallback below instead of claiming a jax backend.
            if is_gpu:
                _ACTIVE_BACKEND = "jax"
                _ARRAY_MODULE = jnp
                _JIT_COMPILER = jax.jit
                _DEVICE_INFO = {
                    "backend": "jax",
                    "platform": platform,
                    "devices": [str(d) for d in devices],
                    "device_count": len(devices),
                    "is_gpu": True,
                    "accelerated": True,
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


def _torch_device() -> Any:
    """Resolve active Torch device (CUDA > MPS > CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_sync() -> None:
    """Block until queued Torch device work completes (guarded no-op on CPU)."""
    try:
        import torch
        dev = _torch_device()
        if dev.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        elif dev.type == "cuda" and hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
    except Exception:
        pass

def _torch_dtype(dtype: Any) -> Any:
    import torch
    resolved = dtype if isinstance(dtype, torch.dtype) else getattr(torch, np.dtype(dtype).name)
    return resolved


def _mlx_dtype(dtype: Any) -> Any:
    import mlx.core as mx
    if isinstance(dtype, mx.Dtype):
        return dtype
    name = np.dtype(dtype).name
    if name == "float64":
        raise TypeError("MLX has no float64; use float32 on Apple Silicon.")
    return getattr(mx, "bool_" if name == "bool" else name)

def to_device(array: Any, dtype: Any = None) -> Any:
    """Transfer to the selected device, preserving resident arrays and integer precision.

    MPS and implicit MLX floating-point inputs use float32 because neither
    supports float64. Explicit float64 requests pass through so callers that
    require it fail loudly instead of silently changing precision. MLX
    supports int64; do not truncate integer data to int32.
    """
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        import torch
        dev = _torch_device()
        if isinstance(array, torch.Tensor):
            if (
                dtype is None
                and dev.type == "mps"
                and array.dtype == torch.float64
            ):
                return array.to(device=dev, dtype=torch.float32)
            return array.to(device=dev, dtype=_torch_dtype(dtype) if dtype is not None else None)
        arr = to_numpy(array)
        if dtype is None and dev.type == "mps" and arr.dtype == np.float64:
            return torch.as_tensor(arr.astype(np.float32), device=dev)
        return torch.as_tensor(arr, dtype=_torch_dtype(dtype) if dtype is not None else None, device=dev)
    if _ACTIVE_BACKEND == "mlx":
        import mlx.core as mx
        target_dtype = _mlx_dtype(dtype) if dtype is not None else None
        if isinstance(array, mx.array):
            return array if target_dtype is None or array.dtype == target_dtype else array.astype(target_dtype)
        arr = to_numpy(array)
        if target_dtype is None and arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        return mx.array(arr, dtype=target_dtype) if target_dtype is not None else mx.array(arr)
    if _ACTIVE_BACKEND == "cupy":
        return xp.asarray(array, dtype=dtype)
    if _ACTIVE_BACKEND == "jax":
        return xp.asarray(array, dtype=dtype)
    return np.ascontiguousarray(to_numpy(array), dtype=dtype)

def to_numpy(tensor: Any) -> np.ndarray:
    """Materialize an array on CPU, even after the active backend has changed."""
    if tensor is None:
        return np.empty((0, 0), dtype=np.float32)
    if isinstance(tensor, np.ndarray):
        return tensor
    module = type(tensor).__module__.split(".")[0]
    if module == "torch":
        return tensor.detach().cpu().numpy()
    if module == "cupy":
        return tensor.get()
    if module == "mlx":
        import mlx.core as mx
        mx.eval(tensor)
        return np.asarray(tensor)
    return np.asarray(tensor)


def zeros(shape: Sequence[int], dtype: Any = np.float32) -> Any:
    """Create zeros on the selected device. Explicit float64 on MPS raises."""
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        return xp.zeros(shape, dtype=_torch_dtype(dtype), device=_torch_device())
    if _ACTIVE_BACKEND == "mlx":
        return xp.zeros(shape, dtype=_mlx_dtype(dtype))
    return xp.zeros(shape, dtype=dtype)


def ones(shape: Sequence[int], dtype: Any = np.float32) -> Any:
    """Create ones on the selected device using its supported dtype."""
    xp = get_array_module()
    if _ACTIVE_BACKEND == "torch":
        return xp.ones(shape, dtype=_torch_dtype(dtype), device=_torch_device())
    if _ACTIVE_BACKEND == "mlx":
        return xp.ones(shape, dtype=_mlx_dtype(dtype))
    return xp.ones(shape, dtype=dtype)


def clip_bounds(x: Any, xl: Any, xu: Any) -> Any:
    """Element-wise box constraint clamping on active device."""
    if "torch" in type(x).__module__:
        import torch
        dev = x.device
        xl_t = xl if isinstance(xl, torch.Tensor) and xl.device == dev else torch.as_tensor(xl, device=dev, dtype=x.dtype)
        xu_t = xu if isinstance(xu, torch.Tensor) and xu.device == dev else torch.as_tensor(xu, device=dev, dtype=x.dtype)
        return torch.clamp(x, xl_t, xu_t)
    if "mlx" in type(x).__module__:
        import mlx.core as mx
        xl_m = xl if "mlx" in type(xl).__module__ else mx.array(xl)
        xu_m = xu if "mlx" in type(xu).__module__ else mx.array(xu)
        return mx.clip(x, xl_m, xu_m)
    xp = get_array_module()
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
