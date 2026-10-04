"""Exact hypervolume (minimisation) in any dimension: sweep for M<=2, WFG-style recursion above."""

from __future__ import annotations

import numpy as np

__all__ = ["hypervolume", "hv_contributions"]


def _nondominated(P: np.ndarray) -> np.ndarray:
    if len(P) < 2:
        return P
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if keep[i]:
            dom = np.all(P[i] <= P, axis=1) & np.any(P[i] < P, axis=1)
            keep &= ~dom
            keep[i] = True
    return P[keep]


def _hv2(P: np.ndarray, ref: np.ndarray) -> float:
    P = P[np.argsort(P[:, 0], kind="stable")]
    hv, prev = 0.0, ref[1]
    for x, y in P:
        if y < prev:
            hv += (ref[0] - x) * (prev - y)
            prev = y
    return float(hv)


def _wfg(P: np.ndarray, ref: np.ndarray) -> float:
    n, m = P.shape
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.prod(ref - P[0]))
    if m == 2:
        return _hv2(P, ref)
    P = P[np.argsort(-P[:, -1], kind="stable")]      # process points with the worst last objective first
    hv = 0.0
    for k in range(n):
        incl = float(np.prod(ref - P[k]))
        if k + 1 < n:
            lim = _nondominated(np.maximum(P[k + 1:], P[k]))
            incl -= _wfg(lim, ref)
        hv += incl
    return hv


def hypervolume(points, ref) -> float:
    """Hypervolume dominated by ``points`` w.r.t. ``ref`` (points not strictly below ``ref`` are ignored)."""
    P = np.atleast_2d(np.asarray(points, dtype=float))
    ref = np.asarray(ref, dtype=float)
    if P.size == 0:
        return 0.0
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    return _wfg(_nondominated(P), ref)


def hv_contributions(points, ref) -> np.ndarray:
    """Exclusive hypervolume contribution of every point: HV(all) - HV(all but i), computed as
    ``prod(ref - p_i) - HV(limit set of the others clipped by p_i)`` (one small recursion per point)."""
    P = np.atleast_2d(np.asarray(points, dtype=float))
    ref = np.asarray(ref, dtype=float)
    out = np.zeros(len(P))
    inside = np.all(P < ref, axis=1)
    for i in np.where(inside)[0]:
        others = np.delete(P, i, axis=0)
        others = others[np.all(others < ref, axis=1)]
        incl = float(np.prod(ref - P[i]))
        if len(others):
            incl -= _wfg(_nondominated(np.maximum(others, P[i])), ref)
        out[i] = incl
    return out
