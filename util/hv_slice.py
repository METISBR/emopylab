"""Weighted hypervolume by objective slicing (HSO-style), with a weight per final slice.

The weight attached to the i-th final slice is what the preference-based indicator of I-SIBEA needs, so the
slicing procedure is implemented as a direct, literal port instead of the WFG recursion in :mod:`util.hv`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cal_whv"]


def _head(pl):
    return None if len(pl) == 0 else pl[0]


def _tail(pl):
    return pl[1:] if len(pl) >= 2 else pl[:0]


def _insert(p, k, pl):
    """Insert ``p`` into the list ``pl`` (sorted on column ``k``) dropping members that ``p`` dominates on k.."""
    ql = []
    flag1 = flag2 = False
    hp = _head(pl)
    while len(pl) and hp[k] < p[k]:
        ql.append(hp)
        pl = _tail(pl)
        hp = _head(pl)
    ql.append(p)
    while len(pl):
        q = _head(pl)
        for i in range(k, len(p)):
            if p[i] < q[i]:
                flag1 = True
            elif p[i] > q[i]:
                flag2 = True
        if not (flag1 and not flag2):
            ql.append(q)
        pl = _tail(pl)
    return np.array(ql) if ql else np.zeros((0, len(p)))


def _key(pl):
    return (pl.shape, pl.tobytes())


def _add(cell, S):
    """Merge ``cell = (weight, point list)`` into the dict ``S`` (keyed by point-list content, insertion ordered)."""
    k = _key(cell[1])
    if k in S:
        S[k] = (S[k][0] + cell[0], S[k][1])
    else:
        S[k] = cell
    return S


def _slice(pl, k, ref):
    p = _head(pl)
    pl = _tail(pl)
    ql = np.zeros((0, len(p)))
    S = {}
    while len(pl):
        ql = _insert(p, k + 1, ql)
        p_ = _head(pl)
        S = _add((abs(p[k] - p_[k]), ql), S)
        p = p_
        pl = _tail(pl)
    ql = _insert(p, k + 1, ql)
    return _add((abs(p[k] - ref[k]), ql), S)


def cal_whv(points, ref, weight=None) -> float:
    P = np.atleast_2d(np.asarray(points, dtype=float))
    ref = np.asarray(ref, dtype=float)
    P = P[~np.any(P > ref, axis=1)]
    if len(P) == 0:
        return 0.0
    M = P.shape[1]
    pl = P[np.lexsort(P.T[::-1])]                                   # sortrows
    S = {_key(pl): (1.0, pl)}
    for k in range(M - 1):
        S_ = {}
        for w, plk in S.values():
            for w2, ql in _slice(plk, k, ref).values():
                S_ = _add((w2 * w, ql), S_)
        S = S_
    score = 0.0
    for i, (w, plk) in enumerate(S.values()):
        p = _head(plk)
        wt = 1.0 if weight is None else float(weight[min(i, len(weight) - 1)])
        score += w * abs(p[M - 1] - ref[M - 1]) * wt
    return score
