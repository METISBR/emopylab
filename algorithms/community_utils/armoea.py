"""Reference-point adaptation and indicator-based selection shared by the AR-MOEA family (distance to adjusted
reference points, reference point update, contribution-based last-front selection)."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import nd_sort
from algorithms.community_utils.robust import cosine_dist

__all__ = ["cal_distance", "update_ref_point", "last_selection", "contribution_fitness"]


def _pdist(A, B):
    return np.sqrt(np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0))


def _rowmax_nan(C):
    """MATLAB ``max(C,[],2)``: NaN entries are ignored (an all-NaN row stays NaN)."""
    allnan = np.all(np.isnan(C), axis=1)
    out = np.max(np.where(np.isnan(C), -np.inf, C), axis=1)
    out[allnan] = np.nan
    return out


def _argmin_nan(v):
    """MATLAB ``[~,x] = min(v)``: NaN ignored, index 0 when everything is NaN."""
    return 0 if np.all(np.isnan(v)) else int(np.argmin(np.where(np.isnan(v), np.inf, v)))


def cal_distance(P, R, clip: bool = True, rng=None):
    """Distance of every solution to every reference point after each reference point has been scaled to the
    projection of the solution nearest to its ray.  ``clip`` floors objectives and references at 1e-6."""
    P, R = np.array(P, dtype=float), np.array(R, dtype=float)
    N, NR = len(P), len(R)
    if clip:
        P, R = np.maximum(P, 1e-6), np.maximum(R, 1e-6)
    elif rng is not None:
        zero = np.where(np.sum(P ** 2, axis=1) == 0)[0]
        if len(zero):
            P[zero] += 1e-6 * rng.random((len(zero), P.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = 1 - cosine_dist(P, R)
        normR, normP = np.sqrt(np.sum(R ** 2, axis=1)), np.sqrt(np.sum(P ** 2, axis=1))
        d1 = normP[:, None] * cosine
        d2 = normP[:, None] * np.sqrt(np.maximum(1 - cosine ** 2, 0))
        d2f = np.where(np.isnan(d2), np.inf, d2)
        nearest = np.argmin(d2f, axis=0)
        R = R * (d1[nearest, np.arange(NR)] / normR)[:, None]
    return _pdist(P, R)


def _pick_far(cosine, choose, limit):
    """Greedily add the unselected item whose largest cosine to the selected ones is smallest, until ``limit``."""
    choose = choose.copy()
    while choose.sum() < limit:
        unsel = np.where(~choose)[0]
        x = _argmin_nan(_rowmax_nan(cosine[np.ix_(~choose, choose)]))
        choose[unsel[x]] = True
    return choose


def update_ref_point(archive, W, rng_=None, clip: bool = True, noise_rng=None):
    """Adapt the reference points to the non-dominated ``archive`` objectives.

    Returns ``(archive, ref_point, range, ratio)`` where ``range`` is ``[ideal; nadir]`` and ``ratio`` the share of the
    original reference points that still attract a solution."""
    archive = np.asarray(archive, dtype=float)
    front, _ = nd_sort(archive, None, 1)
    archive = np.unique(archive[front == 1], axis=0)
    NA, NW = len(archive), len(W)
    rng_ = None if rng_ is None or np.size(rng_) == 0 else np.array(rng_, dtype=float)
    if rng_ is not None:
        rng_[0] = np.minimum(rng_[0], archive.min(axis=0)) if NA else rng_[0]
    elif NA:
        rng_ = np.vstack([archive.min(axis=0), archive.max(axis=0)])
    if NA <= 1:
        return archive, np.asarray(W, dtype=float), rng_, 0.0
    tA = archive - rng_[0]
    W = W * (rng_[1] - rng_[0])
    dist = cal_distance(tA, W, clip, noise_rng)
    nearest_p = np.argmin(dist, axis=0)
    contributing = np.unique(nearest_p)
    nearest_w = np.argmin(dist, axis=1)
    valid_w = np.unique(nearest_w[contributing])
    choose = np.isin(np.arange(NA), contributing)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = 1 - cosine_dist(tA, tA)
    np.fill_diagonal(cos, 0.0)
    choose = _pick_far(cos, choose, min(3 * NW, NA))
    archive, tA = archive[choose], tA[choose]
    ref = np.vstack([W[valid_w], tA])
    choose = np.concatenate([np.ones(len(valid_w), bool), np.zeros(len(tA), bool)])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = 1 - cosine_dist(ref, ref)
    np.fill_diagonal(cos, 0.0)
    choose = _pick_far(cos, choose, min(NW, len(ref)))
    return archive, ref[choose], rng_, len(valid_w) / NW


def contribution_fitness(F, ref, rng_, clip: bool = True, noise_rng=None):
    """Contribution of every solution to the indicator (larger is better; non-contributing solutions are penalised by
    their convergence only)."""
    N = len(F)
    dist = cal_distance(F - rng_[0], ref, clip, noise_rng)
    conv = dist.min(axis=1)
    rank = np.argsort(dist, axis=0, kind="stable")
    dis = np.take_along_axis(dist, rank, axis=0)
    non = np.ones(N, bool)
    non[rank[0]] = False
    metric = dis[0].sum() + conv[non].sum()
    fit = np.full(N, np.inf)
    fit[non] = metric - conv[non]
    for p in np.where(~non)[0]:
        t = rank[0] == p
        nc = np.zeros(N, bool)
        nc[rank[1][t]] = True
        nc &= non
        fit[p] = metric - dis[0][t].sum() + dis[1][t].sum() - conv[nc].sum()
    return fit


def last_selection(F, ref, rng_, K, clip: bool = True, noise_rng=None):
    """Boolean mask of the ``K`` solutions kept: the solution whose removal hurts the indicator least is deleted."""
    N, NR = len(F), len(ref)
    dist = cal_distance(F - rng_[0], ref, clip, noise_rng)
    conv = dist.min(axis=1)
    rank = np.argsort(dist, axis=0, kind="stable")
    dis = np.take_along_axis(dist, rank, axis=0)
    remain = np.ones(N, bool)
    while remain.sum() > K:
        non = remain.copy()
        non[rank[0]] = False
        metric = dis[0].sum() + conv[non].sum()
        m = np.full(N, np.inf)
        m[non] = metric - conv[non]
        for p in np.where(remain & ~non)[0]:
            t = rank[0] == p
            nc = np.zeros(N, bool)
            nc[rank[1][t]] = True
            nc &= non
            m[p] = metric - dis[0][t].sum() + dis[1][t].sum() - conv[nc].sum()
        d = int(np.argmin(m))
        keep = rank != d                               # exactly one entry per column
        dis = dis.T[keep.T].reshape(NR, -1).T
        rank = rank.T[keep.T].reshape(NR, -1).T
        remain[d] = False
    return remain
