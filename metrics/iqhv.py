"""Improved Quick Hypervolume (IQHV / QHV-II) exact algorithm.

Reference:
  A. Jaszkiewicz. Improved quick hypervolume algorithm.
  Computers & Operations Research 90 (2018) 72-83.
  DOI: 10.1016/j.cor.2017.09.016

Algorithm (Sec. 3, Alg. 1 + QHV-II splitting scheme), maximization form:
  Work in X-space: X_i = ref - F_i, lo = zeros, hi = max(X).
  Dominated volume of X_i w.r.t. lo is prod(X_i - lo).
  - Pivot: point maximizing prod(min(pts,hi) - lo).
  - QHV-II split into d sub-problems (d unions of basic hypercuboids):
      H_1: s_1 >= p_1
      H_j: s_l < p_l (l<j) AND s_j >= p_j   (j = 2..d)
    Points projected (clamped) onto H_j, filtered to non-dominated,
    recursion on (proj, lo_j, hi_j).
  - Base cases: 0 pts -> 0; 1 pt -> box; 2 pts -> inclusion-exclusion.

Backend: irregular pivot-dependent recursion tree -> pure NumPy float64
recursion (correctness oracle). Per-node ops vectorized. Exact ground
truth for small N / low M; oracle for MC / R2 / ND-Tree estimators.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _box_volume(lo: np.ndarray, hi: np.ndarray) -> float:
    side = hi - lo
    if np.any(side <= 0.0):
        return 0.0
    return float(np.prod(side))


def _filter_dominated_max(pts: np.ndarray) -> np.ndarray:
    """Keep non-dominated points (maximization)."""
    n = pts.shape[0]
    if n <= 1:
        return pts
    ge = np.all(pts[None, :, :] >= pts[:, None, :], axis=-1)
    gt = np.any(pts[None, :, :] > pts[:, None, :], axis=-1)
    dom = ge & gt
    np.fill_diagonal(dom, False)
    return pts[~np.any(dom, axis=1)]


def _hv_recursive(pts: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    n = pts.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return _box_volume(lo, np.minimum(pts[0], hi))
    if n == 2:
        c0 = np.minimum(pts[0], hi)
        c1 = np.minimum(pts[1], hi)
        v0 = _box_volume(lo, c0)
        v1 = _box_volume(lo, c1)
        return float(v0 + v1 - _box_volume(lo, np.minimum(c0, c1)))

    corners = np.minimum(pts, hi)
    vols = np.prod(np.maximum(corners - lo, 0.0), axis=1)
    best = int(np.argmax(vols))
    pivot = corners[best]
    total = float(np.prod(np.maximum(pivot - lo, 0.0)))
    if total <= 0.0:
        return 0.0

    d = pts.shape[1]
    rest = np.delete(pts, best, axis=0)
    for j in range(d):
        lo_j = lo.copy()
        lo_j[j] = pivot[j]
        hi_j = hi.copy()
        for ell in range(j):
            hi_j[ell] = pivot[ell]
        if np.any(hi_j <= lo_j):
            continue
        proj = np.minimum(rest, hi_j)
        valid = np.all(proj > lo_j, axis=1)
        if not np.any(valid):
            continue
        sub = _filter_dominated_max(proj[valid])
        if sub.shape[0] == 0:
            continue
        total += _hv_recursive(sub, lo_j, hi_j)
    return float(total)


def iqhv(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact hypervolume of F (minimization) w.r.t. ref_point via QHV-II."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    ref = np.asarray(ref_point, dtype=np.float64).reshape(-1)
    if F.size == 0 or ref.size == 0 or F.shape[1] != ref.shape[0]:
        return 0.0
    valid = np.all(F <= ref, axis=1)
    pts = F[valid]
    if pts.shape[0] == 0:
        return 0.0
    X = ref - pts  # maximization space, lo = 0
    lo = np.zeros(ref.shape[0])
    hi = np.max(X, axis=0)
    Xnd = _filter_dominated_max(np.ascontiguousarray(X))
    return float(_hv_recursive(Xnd, lo, hi.copy()))


def iqhv_contributions(F: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    """Exclusive HV contribution of each point (leave-one-out, exact)."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    total = iqhv(F, ref_point)
    out = np.zeros(F.shape[0], dtype=float)
    for i in range(F.shape[0]):
        out[i] = total - iqhv(np.delete(F, i, axis=0), ref_point)
    return out


class IQHV:
    """Indicator class wrapper matching emopylab Indicator protocol."""

    def __init__(self, ref_point: np.ndarray, **kwargs: Any) -> None:
        self.ref_point = np.asarray(ref_point, dtype=float)
        self.kwargs = dict(kwargs)

    def do(self, F: np.ndarray) -> float:
        return float(iqhv(F, self.ref_point))


def _emopylab_wrapper(front: np.ndarray, context: dict) -> float:
    ctx = dict(context or {})
    F = np.atleast_2d(np.asarray(front, dtype=float))
    ref = ctx.get("ref_point")
    if ref is None:
        pf = ctx.get("pareto_front")
        ref = np.ones(F.shape[1]) if pf is None else np.max(np.atleast_2d(np.asarray(pf, dtype=float)), axis=0) * 1.1
    return float(iqhv(F, np.asarray(ref, dtype=float)))


METRICS = {
    "IQHV": _emopylab_wrapper,
}
