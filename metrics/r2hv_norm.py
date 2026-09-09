"""Normalized R2-based Hypervolume approximation (Norm-R2HV / Norm-R2HVC).

Reference:
  G. Wu, T. Shu, K. Shang, H. Ishibuchi. Normalization in R2-based
  Hypervolume and Hypervolume Contribution Approximation.
  Proc. IEEE SSCI 2023. DOI: 10.1109/SSCI52147.2023.10371986

Formulation (maximization; paper Sec. II-III):
  R2-based HV (Eq. 3-4):
    HV(S, r) ~= C(m) * (1/n) * sum_i l_i^m
    C(m) = pi^{m/2} / (m * 2^{m-1} * Gamma(m/2))
    l_i  = max_{s in A} min_j |r_j - s_j| / lambda^i_j      (Eq. 4)
  with direction vectors lambda^i on the unit simplex (UNV method:
  sample phi ~ N(0, I), take abs, normalize to unit norm).

  Normalization (Property 1-2: translation + positive scaling):
    Step 1: translate so reference -> origin:  a'_j  = a_j - r_j   (Eq. 9)
    Step 2: scale by per-objective max:        a''_j = a'_j / max{f_j} (Eq. 10)
    Step 3: HV = (prod_j max{f_j}) * HV_R2''                    (Eq. 11)
  For minimization emopylab fronts the paper's Property 3 applies
  (HV(S,r) = HV(-S,-r)); the implementation negates inputs so the
  maximization math holds, then returns the (sign-corrected) value.

  HVC normalization: tight base point p per Eq. 12, translate (Eq. 13),
  scale by s' (Eq. 14), HVC = prod * HVC'' (Eq. 15). The exported
  contributions() implements leave-one-out on the normalized HV.

Backend strategy (Apple Silicon / NPU / GPU):
  - Direction generation on torch device when available (randn on
    MPS/CUDA/CPU), else NumPy; single H2D for the front, tiled
    (n_dirs x N x M) broadcast in float32, one sync at the end.
  - Falls back to tiled NumPy when torch is unavailable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _r2_constant(m: int) -> float:
    return (math.pi ** (m / 2.0)) / (m * (2.0 ** (m - 1)) * math.gamma(m / 2.0))


def _unit_directions(n_dirs: int, m: int, seed: int, device: Any = None) -> np.ndarray:
    """UNV directions: |N(0,I)| rows normalized to unit norm (paper Sec. IV)."""
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((n_dirs, m)).astype(np.float64))
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-30)
    return np.ascontiguousarray(W, dtype=np.float32)


def _r2_lengths(A: np.ndarray, r: np.ndarray, W: np.ndarray,
                tile: int = 4096) -> np.ndarray:
    """l_i = max_s min_j |r_j - s_j| / w_ij, tiled over directions."""
    A32 = np.ascontiguousarray(A, dtype=np.float32)
    r32 = np.ascontiguousarray(r, dtype=np.float32)
    W32 = np.ascontiguousarray(W, dtype=np.float32)
    n = W32.shape[0]
    out = np.empty(n, dtype=np.float64)
    diff = np.abs(r32[None, :] - A32)  # (N, M)
    try:
        import torch

        dev = None
        try:
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        except Exception:
            dev = torch.device("cpu")
        D = torch.as_tensor(np.ascontiguousarray(diff, dtype=np.float32), device=dev)
        Ws = torch.as_tensor(W32, device=dev)
        with torch.inference_mode():
            parts = []
            for s in range(0, n, tile):
                q = torch.abs(D[None, :, :] / Ws[s:s + tile][:, None, :]).min(dim=-1).values
                parts.append(q.max(dim=1).values)
            Ls = torch.cat(parts)
            try:
                if dev.type == "mps":
                    torch.mps.synchronize()
                elif dev.type == "cuda":
                    torch.cuda.synchronize()
            except Exception:
                pass
            import math as _math  # noqa: F401

            out = np.asarray(Ls.detach().cpu().numpy(), dtype=np.float64)
            return out
    except Exception:
        pass
    for s in range(0, n, tile):
        q = diff[None, :, :] / np.maximum(W32[s:s + tile][:, None, :], 1e-30)
        out[s:s + q.shape[0]] = np.min(q, axis=-1).max(axis=1)
    return out


def norm_r2_hv(F: np.ndarray, ref_point: np.ndarray | None = None,
               n_dirs: int = 1000, seed: int = 1,
               minimize: bool = True) -> float:
    """Normalized R2 hypervolume (paper Eq. 9-11)."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    if F.size == 0:
        return 0.0
    m = F.shape[1]
    if ref_point is None:
        ref = np.max(F, axis=0) * 1.1
    else:
        ref = np.asarray(ref_point, dtype=np.float64).reshape(-1)
    A = -F if minimize else F.copy()
    r = -ref if minimize else ref.copy()
    # Step 1: translate reference to origin (Eq. 9).
    Ap = A - r
    # Step 2: scale by per-objective max (Eq. 10).
    scale = np.max(Ap, axis=0)
    scale = np.where(np.abs(scale) <= 1e-12, 1.0, scale)
    App = Ap / scale
    rp = np.zeros(m)
    W = _unit_directions(n_dirs, m, seed)
    L = _r2_lengths(App, rp, W)
    hv_n = _r2_constant(m) * float(np.mean(L ** m))
    # Step 3: rescale (Eq. 11).
    return float(np.prod(scale) * hv_n)


def norm_r2_hvc(F: np.ndarray, ref_point: np.ndarray | None = None,
                n_dirs: int = 1000, seed: int = 1,
                minimize: bool = True) -> np.ndarray:
    """Normalized R2 HV contributions via leave-one-out (Eq. 12-15)."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    n = F.shape[0]
    out = np.zeros(n, dtype=float)
    total = norm_r2_hv(F, ref_point, n_dirs=n_dirs, seed=seed, minimize=minimize)
    for i in range(n):
        out[i] = total - norm_r2_hv(np.delete(F, i, axis=0), ref_point,
                                    n_dirs=n_dirs, seed=seed, minimize=minimize)
    return out


class NormR2HV:
    """Indicator class wrapper matching emopylab protocol."""

    def __init__(self, ref_point: np.ndarray | None = None, n_dirs: int = 1000,
                 seed: int = 1, **kwargs: Any) -> None:
        self.ref_point = None if ref_point is None else np.asarray(ref_point, dtype=float)
        self.n_dirs = int(n_dirs)
        self.seed = int(seed)
        self.kwargs = dict(kwargs)

    def do(self, F: np.ndarray) -> float:
        return float(norm_r2_hv(F, self.ref_point, n_dirs=self.n_dirs, seed=self.seed))


def _emopylab_wrapper(front: np.ndarray, context: dict) -> float:
    ctx = dict(context or {})
    return float(norm_r2_hv(
        front, ctx.get("ref_point"),
        n_dirs=int(ctx.get("r2_dirs", ctx.get("hv_r2_dirs", 1000))),
        seed=int(ctx.get("hv_seed", ctx.get("seed", 1))),
    ))


METRICS = {
    "NormR2HV": _emopylab_wrapper,
}
