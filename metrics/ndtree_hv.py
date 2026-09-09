"""ND-Tree distance-based Hypervolume estimation (HVE_ND-Tree_New).

Reference:
  A. Jaszkiewicz, P. Zielniewicz. Improving the Efficiency of the
  Distance-Based Hypervolume Estimation Using ND-Tree.
  IEEE Trans. Evolutionary Computation 29(3), June 2025.
  DOI: 10.1109/TEVC.2024.3391857

Formulation (paper Sec. III, Deng-Zhang polar integration):
  Work in X-space: X_i = ref - F_i (maximization, lo = 0).
  HV = C(d) * E_psi[L(psi)^d],  C(d) = 2*pi^{d/2} / (d^2 * Gamma(d/2))
  L(psi) = max_{z in S} min_j z_j / psi_j   (ray-box hit distance),
  psi uniform on the positive unit sphere (unit normal vector approach:
  phi ~ N(0,I), psi = |phi|/|phi|).
  ND-Tree (Alg. 1): branch-and-bound max search using per-node local
  ideal points: prune subtree when min_j ideal_j / psi_j <= best
  (monotonicity, cf. Eq. 12-13). Tree build (Alg. 2-4): furthest-point
  seeding into <= d+1 children, lexicographic closeness to local ideal,
  leaf capacity 20.

Backend strategy (Apple Silicon / NPU / GPU):
  - Exact same estimator values as exhaustive search for the same seed;
    ND-Tree only prunes.
  - Vectorized NumPy per-node bounds in float64; leaf scan vectorized.
  - Torch fast path (MPS/CUDA/CPU, float32, tiled, one sync) for the
    exhaustive branch used at small N; ND-Tree path for large N.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class _Node:
    __slots__ = ("ideal", "children", "points", "is_leaf")

    def __init__(self, ideal: np.ndarray, points: np.ndarray | None = None,
                 children: list | None = None):
        self.ideal = ideal
        self.points = points
        self.children = children if children is not None else []
        self.is_leaf = points is not None


def _update_ideal(node_ideal: np.ndarray, y: np.ndarray) -> bool:
    mask = y < node_ideal
    if np.any(mask):
        node_ideal[mask] = y[mask]
        return True
    return False


def _split_points(pts: np.ndarray, max_children: int) -> list[np.ndarray]:
    """Furthest-point seeding (Alg. 3)."""
    n = pts.shape[0]
    if n <= max_children:
        return [pts[i:i + 1] for i in range(n)]
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    mean_d = D.mean(axis=1)
    seeds: list[int] = [int(np.argmax(mean_d))]
    while len(seeds) < max_children:
        rest = [i for i in range(n) if i not in seeds]
        score = D[np.ix_(rest, seeds)].mean(axis=1)
        seeds.append(rest[int(np.argmax(score))])
    groups: list[list[int]] = [[s] for s in seeds]
    assigned = set(seeds)
    ideals = [pts[s].copy() for s in seeds]
    for i in range(n):
        if i in assigned:
            continue
        best, best_key = 0, None
        for c, ideal in enumerate(ideals):
            dom = bool(np.all(ideal <= pts[i]))
            key = (0 if dom else 1, float(np.linalg.norm(pts[i] - ideal)))
            if best_key is None or key < best_key:
                best, best_key = c, key
        groups[best].append(i)
        _update_ideal(ideals[best], pts[i])
    return [pts[np.array(g)] for g in groups]


def _build_tree(pts: np.ndarray, max_children: int, leaf_cap: int = 20) -> _Node:
    ideal = np.min(pts, axis=0)
    if pts.shape[0] <= leaf_cap:
        return _Node(ideal, points=np.ascontiguousarray(pts))
    node = _Node(ideal.copy(), children=[])
    for grp in _split_points(pts, max_children):
        child = _build_tree(grp, max_children, leaf_cap)
        node.children.append(child)
        _update_ideal(node.ideal, child.ideal)
    return node


def _bound(pts: np.ndarray, psi: np.ndarray) -> float:
    return float(np.max(np.min(pts / np.maximum(psi, 1e-300), axis=1)))


def _search(node: _Node, psi: np.ndarray, best: float) -> float:
    s_ideal = float(np.min(node.ideal / np.maximum(psi, 1e-300)))
    if s_ideal <= best:
        return best
    if node.is_leaf:
        assert node.points is not None
        return max(best, _bound(node.points, psi))
    order = sorted(node.children,
                   key=lambda c: float(np.min(c.ideal / np.maximum(psi, 1e-300))),
                   reverse=True)
    for c in order:
        if float(np.min(c.ideal / np.maximum(psi, 1e-300))) > best:
            best = _search(c, psi, best)
    return best


def _estimate_ndtree(S: np.ndarray, Psi: np.ndarray, leaf_cap: int = 20) -> np.ndarray:
    """Boundary distance per direction via ND-Tree branch-and-bound."""
    tree = _build_tree(np.ascontiguousarray(S, dtype=np.float64),
                       max_children=S.shape[1] + 1, leaf_cap=leaf_cap)
    out = np.empty(Psi.shape[0], dtype=np.float64)
    for i, psi in enumerate(Psi):
        out[i] = _search(tree, psi, best=float("-inf"))
    return out


def _estimate_exhaustive(S: np.ndarray, Psi: np.ndarray,
                         tile: int = 2048) -> np.ndarray:
    """Exhaustive boundary distance: same values, no pruning."""
    S64 = np.ascontiguousarray(S, dtype=np.float64)
    n_dirs = Psi.shape[0]
    out = np.empty(n_dirs, dtype=np.float64)
    try:
        import torch

        dev = torch.device("cpu")
        try:
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = torch.device("mps")
        except Exception:
            pass
        Sd = torch.as_tensor(np.ascontiguousarray(S64, dtype=np.float32), device=dev)
        with torch.inference_mode():
            for s in range(0, n_dirs, tile):
                P = torch.as_tensor(
                    np.ascontiguousarray(Psi[s:s + tile], dtype=np.float32), device=dev)
                Pc = torch.clamp(P, min=1e-30)
                vals = (Sd[None, :, :] / Pc[:, None, :]).min(dim=-1).values
                best = vals.max(dim=1).values
                out[s:s + best.shape[0]] = np.asarray(
                    best.detach().cpu().numpy(), dtype=np.float64)
            try:
                if dev.type == "mps":
                    torch.mps.synchronize()
                elif dev.type == "cuda":
                    torch.cuda.synchronize()
            except Exception:
                pass
        return out
    except Exception:
        pass
    for s in range(0, n_dirs, tile):
        P = np.maximum(Psi[s:s + tile], 1e-300)
        out[s:s + P.shape[0]] = np.max(
            np.min(S64[None, :, :] / P[:, None, :], axis=-1), axis=1)
    return out


def _directions(n_dirs: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Phi = np.abs(rng.standard_normal((n_dirs, d)))
    Phi /= np.maximum(np.linalg.norm(Phi, axis=1, keepdims=True), 1e-300)
    return np.ascontiguousarray(Phi, dtype=np.float64)


def _sphere_const(d: int) -> float:
    # R2 polar constant (Shang/Ishibuchi Eq. 3; Deng-Zhang polar form):
    # HV = pi^{d/2} / (d * 2^{d-1} * Gamma(d/2)) * E[L^d].
    return (math.pi ** (d / 2.0)) / (d * (2.0 ** (d - 1)) * math.gamma(d / 2.0))




def ndtree_hv(F: np.ndarray, ref_point: np.ndarray | None = None,
              n_dirs: int = 100_000, seed: int = 1,
              minimize: bool = True, use_tree: bool = True,
              leaf_cap: int = 20) -> float:
    """ND-Tree distance-based HV estimate (X-space polar integration)."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    if F.size == 0:
        return 0.0
    d = F.shape[1]
    if ref_point is None:
        ref = np.max(F, axis=0) * 1.1
    else:
        ref = np.asarray(ref_point, dtype=np.float64).reshape(-1)
    X = ref - F  # maximization space, lo = 0
    X = X[np.all(X >= 0.0, axis=1)]
    if X.shape[0] == 0:
        return 0.0
    Psi = _directions(n_dirs, d, seed)
    if use_tree and X.shape[0] > 2000:
        L = _estimate_ndtree(X, Psi, leaf_cap=leaf_cap)
    else:
        L = _estimate_exhaustive(X, Psi)
    return float(_sphere_const(d) * float(np.mean(L ** d)))


class NDTreeHV:
    """Indicator class wrapper matching emopylab protocol."""

    def __init__(self, ref_point: np.ndarray | None = None,
                 n_dirs: int = 100_000, seed: int = 1, **kwargs: Any) -> None:
        self.ref_point = None if ref_point is None else np.asarray(ref_point, dtype=float)
        self.n_dirs = int(n_dirs)
        self.seed = int(seed)
        self.kwargs = dict(kwargs)

    def do(self, F: np.ndarray) -> float:
        return float(ndtree_hv(F, self.ref_point, n_dirs=self.n_dirs, seed=self.seed))


def _emopylab_wrapper(front: np.ndarray, context: dict) -> float:
    ctx = dict(context or {})
    return float(ndtree_hv(
        front, ctx.get("ref_point"),
        n_dirs=int(ctx.get("hv_nd_dirs", ctx.get("hv_mc_samples", 10_000))),
        seed=int(ctx.get("hv_seed", ctx.get("seed", 1))),
    ))


METRICS = {
    "NDTreeHV": _emopylab_wrapper,
}
