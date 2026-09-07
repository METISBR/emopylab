"""EmoPyLab Bitwise Boolean Matrix Non-Dominated Sorting for GPU/SIMD."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.tensor.backend import get_array_module, to_device, to_numpy


def boolean_matrix_nds(F_tensor: Any, CV_tensor: Any = None) -> list[np.ndarray]:
    """Computes Pareto dominance fronts using vectorized tensor broadcast operations in O(N^2).

    Supports PyTorch (CUDA / ROCm / MPS), Apple MLX, and NumPy.
    Optionally enforces Deb's Constrained Dominance Principle (CDP) if CV_tensor is provided:
      1. Feasible dominates Infeasible.
      2. Infeasible with smaller constraint violation dominates.
      3. For mutually feasible solutions, Pareto dominance applies.
    """
    N = F_tensor.shape[0]
    if N == 0:
        return []
    if N == 1:
        return [np.array([0], dtype=np.int64)]

    # 1. Detect tensor framework and compute Pareto dominance matrix on device
    is_torch = "torch" in type(F_tensor).__module__
    is_mlx = "mlx" in type(F_tensor).__module__
    is_cupy = "cupy" in type(F_tensor).__module__
    if is_torch:
        import torch
        F = F_tensor
        le = F.unsqueeze(1) <= F.unsqueeze(0)
        lt = F.unsqueeze(1) < F.unsqueeze(0)
        dom_matrix = torch.all(le, dim=-1) & torch.any(lt, dim=-1)

        if CV_tensor is not None:
            cv = CV_tensor.squeeze(-1) if CV_tensor.ndim > 1 else CV_tensor
            cv_i = cv.unsqueeze(1)
            cv_j = cv.unsqueeze(0)
            feas_i = cv_i <= 1e-8
            feas_j = cv_j <= 1e-8
            dom_feas = feas_i & (~feas_j)
            dom_infeas = (~feas_i) & (~feas_j) & (cv_i < cv_j)
            dom_obj = feas_i & feas_j & dom_matrix
            dom_matrix = dom_feas | dom_infeas | dom_obj

        dom_np = dom_matrix.detach().cpu().numpy()

    elif is_mlx:
        import mlx.core as mx
        F = F_tensor
        le = mx.expand_dims(F, 1) <= mx.expand_dims(F, 0)
        lt = mx.expand_dims(F, 1) < mx.expand_dims(F, 0)
        dom_matrix = mx.all(le, axis=-1) & mx.any(lt, axis=-1)

        if CV_tensor is not None:
            cv = CV_tensor.squeeze(-1) if CV_tensor.ndim > 1 else CV_tensor
            cv_i = mx.expand_dims(cv, 1)
            cv_j = mx.expand_dims(cv, 0)
            feas_i = cv_i <= 1e-8
            feas_j = cv_j <= 1e-8
            dom_feas = feas_i & (~feas_j)
            dom_infeas = (~feas_i) & (~feas_j) & (cv_i < cv_j)
            dom_obj = feas_i & feas_j & dom_matrix
            dom_matrix = dom_feas | dom_infeas | dom_obj

        dom_np = np.array(dom_matrix)
    elif is_cupy:
        import cupy as cp
        F = F_tensor
        le = F[:, None, :] <= F[None, :, :]
        lt = F[:, None, :] < F[None, :, :]
        dom_matrix = cp.all(le, axis=-1) & cp.any(lt, axis=-1)

        if CV_tensor is not None:
            cv = CV_tensor.squeeze(-1) if CV_tensor.ndim > 1 else CV_tensor
            cv_i = cv[:, None]
            cv_j = cv[None, :]
            feas_i = cv_i <= 1e-8
            feas_j = cv_j <= 1e-8
            dom_feas = feas_i & (~feas_j)
            dom_infeas = (~feas_i) & (~feas_j) & (cv_i < cv_j)
            dom_obj = feas_i & feas_j & dom_matrix
            dom_matrix = dom_feas | dom_infeas | dom_obj

        dom_np = cp.asnumpy(dom_matrix)

    else:
        F = np.asarray(to_numpy(F_tensor), dtype=np.float32)
        le = F[:, None, :] <= F[None, :, :]
        lt = F[:, None, :] < F[None, :, :]
        dom_matrix = np.all(le, axis=-1) & np.any(lt, axis=-1)

        if CV_tensor is not None:
            cv = np.asarray(to_numpy(CV_tensor), dtype=np.float32).ravel()
            cv_i = cv[:, None]
            cv_j = cv[None, :]
            feas_i = cv_i <= 1e-8
            feas_j = cv_j <= 1e-8
            dom_feas = feas_i & (~feas_j)
            dom_infeas = (~feas_i) & (~feas_j) & (cv_i < cv_j)
            dom_obj = feas_i & feas_j & dom_matrix
            dom_matrix = dom_feas | dom_infeas | dom_obj

        dom_np = dom_matrix

    # 2. Front peeling using the precomputed boolean dominance matrix
    domination_counts = np.sum(dom_np, axis=0)  # how many individuals dominate j
    dominated_sets = [np.where(dom_np[i, :])[0] for i in range(N)]

    fronts = []
    current_front = np.where(domination_counts == 0)[0]

    while len(current_front) > 0:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = np.array(next_front, dtype=np.int64)

    return fronts
