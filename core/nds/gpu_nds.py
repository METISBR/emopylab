"""EmoPyLab Bitwise Boolean Matrix Non-Dominated Sorting for GPU/SIMD."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.tensor.backend import to_numpy

_TILE_BUDGET = 2 ** 26


def _numpy_dom_matrix(F: np.ndarray, CV: np.ndarray | None, feas_tol: float = 1e-8) -> np.ndarray:
    le = F[:, None, :] <= F[None, :, :]
    lt = F[:, None, :] < F[None, :, :]
    dom = np.all(le, axis=-1) & np.any(lt, axis=-1)
    if CV is not None:
        cv = np.asarray(CV, dtype=np.float32).ravel()
        cv_i = cv[:, None]
        cv_j = cv[None, :]
        feas_i = cv_i <= feas_tol
        feas_j = cv_j <= feas_tol
        dom = (feas_i & (~feas_j)) | ((~feas_i) & (~feas_j) & (cv_i < cv_j)) | (feas_i & feas_j & dom)
    return dom


def boolean_matrix_nds(
    F_tensor: Any,
    CV_tensor: Any = None,
    feas_tol: float = 1e-8,
) -> list[np.ndarray]:
    """Computes Pareto dominance fronts using vectorized tensor broadcast operations.

    Front membership matches Deb's fast sort exactly for unconstrained inputs; indices
    within a front are ascending (np.where order). Constrained inputs use CDP with
    feas_tol (default 1e-8); penalty/tie-break wrappers (fast sort, ENS,
    find_non_dominated, CDP equal-CV, SDR) intentionally keep their own semantics.
    Supports PyTorch (CUDA / ROCm / MPS), Apple MLX, CuPy CUDA, and NumPy.
    Any device failure falls back to the NumPy CPU path with identical semantics.
    """
    N = int(F_tensor.shape[0])
    if N == 0:
        return []
    if N == 1:
        return [np.array([0], dtype=np.int64)]

    M = int(F_tensor.shape[1])
    if M == 0:
        raise ValueError("Objective matrix F must have at least one column.")

    dom_np: np.ndarray | None = None
    mod = type(F_tensor).__module__
    try:
        if "torch" in mod:
            import torch

            F = F_tensor
            block_size = max(1, min(N, _TILE_BUDGET // max(1, N * M)))
            dom_rows = []
            for start in range(0, N, block_size):
                chunk = F[start:start + block_size]
                le = chunk.unsqueeze(1) <= F.unsqueeze(0)
                lt = chunk.unsqueeze(1) < F.unsqueeze(0)
                dom_chunk = torch.all(le, dim=-1) & torch.any(lt, dim=-1)
                if CV_tensor is not None:
                    cv = CV_tensor.squeeze(-1) if getattr(CV_tensor, "ndim", 1) > 1 else CV_tensor
                    cv_chunk = cv[start:start + block_size].unsqueeze(1)
                    cv_all = cv.unsqueeze(0)
                    feas_i = cv_chunk <= feas_tol
                    feas_j = cv_all <= feas_tol
                    dom_chunk = (feas_i & (~feas_j)) | ((~feas_i) & (~feas_j) & (cv_chunk < cv_all)) | (feas_i & feas_j & dom_chunk)
                dom_rows.append(dom_chunk.detach().cpu())
            dom_np = torch.cat(dom_rows, dim=0).numpy()
        elif "mlx" in mod:
            import mlx.core as mx

            F = F_tensor
            block_size = max(1, min(N, _TILE_BUDGET // max(1, N * M)))
            dom_rows = []
            for start in range(0, N, block_size):
                chunk = F[start:start + block_size]
                le = mx.expand_dims(chunk, 1) <= mx.expand_dims(F, 0)
                lt = mx.expand_dims(chunk, 1) < mx.expand_dims(F, 0)
                dom_chunk = mx.all(le, axis=-1) & mx.any(lt, axis=-1)
                if CV_tensor is not None:
                    cv = CV_tensor.squeeze(-1) if getattr(CV_tensor, "ndim", 1) > 1 else CV_tensor
                    cv_chunk = mx.expand_dims(cv[start:start + block_size], 1)
                    cv_all = mx.expand_dims(cv, 0)
                    feas_i = cv_chunk <= feas_tol
                    feas_j = cv_all <= feas_tol
                    dom_chunk = (feas_i & (~feas_j)) | ((~feas_i) & (~feas_j) & (cv_chunk < cv_all)) | (feas_i & feas_j & dom_chunk)
                mx.eval(dom_chunk)
                dom_rows.append(np.asarray(dom_chunk))
            dom_np = np.concatenate(dom_rows, axis=0)
        elif "cupy" in mod:
            import cupy as cp

            F = F_tensor
            block_size = max(1, min(N, _TILE_BUDGET // max(1, N * M)))
            dom_rows = []
            for start in range(0, N, block_size):
                chunk = F[start:start + block_size]
                le = chunk[:, None, :] <= F[None, :, :]
                lt = chunk[:, None, :] < F[None, :, :]
                dom_chunk = cp.all(le, axis=-1) & cp.any(lt, axis=-1)
                if CV_tensor is not None:
                    cv = CV_tensor.squeeze(-1) if getattr(CV_tensor, "ndim", 1) > 1 else CV_tensor
                    cv_chunk = cv[start:start + block_size, None]
                    cv_all = cv[None, :]
                    feas_i = cv_chunk <= feas_tol
                    feas_j = cv_all <= feas_tol
                    dom_chunk = (feas_i & (~feas_j)) | ((~feas_i) & (~feas_j) & (cv_chunk < cv_all)) | (feas_i & feas_j & dom_chunk)
                dom_rows.append(cp.asnumpy(dom_chunk))
            dom_np = np.concatenate(dom_rows, axis=0)
    except Exception:
        dom_np = None

    if dom_np is None:
        F = np.asarray(to_numpy(F_tensor), dtype=np.float32)
        CV = np.asarray(to_numpy(CV_tensor), dtype=np.float32).ravel() if CV_tensor is not None else None
        dom_np = _numpy_dom_matrix(F, CV, feas_tol=feas_tol)

    # 2. Front peeling on vectorized counts (no per-row Python lists).
    domination_counts = dom_np.sum(axis=0).astype(np.int64)
    remaining = np.ones(N, dtype=bool)
    fronts: list[np.ndarray] = []
    while True:
        current_front = np.where((domination_counts == 0) & remaining)[0]
        if len(current_front) == 0:
            break
        current_front = current_front.astype(np.int64)
        fronts.append(current_front)
        remaining[current_front] = False
        if not remaining.any():
            break
        domination_counts -= dom_np[current_front].sum(axis=0)
        domination_counts[~remaining] = -1

    return fronts
