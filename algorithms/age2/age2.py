"""EmoPyLab Native AGE-MOEA-II (Approximation-Guided Evolutionary MOEA II).

Reference:
A. Panichella. An improved Pareto front modeling algorithm for large-scale many-objective optimization. Proceedings of
the Genetic and Evolutionary Computation Conference, 2022, 565-573.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament
from core.population import Population

__all__ = [
    "AGEMOEA2",
]


def _lp(X, p):
    with np.errstate(all="ignore"):
        return np.sum(np.abs(X) ** p, axis=-1) ** (1.0 / p)


def _newton(point, precision=0.001):
    """Curvature ``x`` with ||point||_x = 1 by Newton-Raphson on log(sum(point^x))."""
    x, prev = 1.0, 1.0
    nz = point != 0
    with np.errstate(all="ignore"):
        for _ in range(100):
            f = np.log(np.sum(point ** x))
            ff = np.sum(point[nz] ** x * np.log(point[nz])) / np.sum(point[nz] ** x)
            x = x - f / ff
            if abs(x - prev) <= precision:
                break
            prev = x
    return float(np.real(x))


def _corners(front):
    m, n = front.shape
    if m <= n:
        return np.arange(m)
    idx = np.zeros(n, dtype=int)
    with np.errstate(all="ignore"):
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1
            d = np.linalg.norm(front - (front @ e)[:, None] * e, axis=1)
            idx[i] = int(np.nanargmin(d)) if not np.all(np.isnan(d)) else 0
    return idx


def _geometry(front):
    d = np.linalg.norm(front, axis=1)
    d[_corners(front)] = np.inf
    point = front[int(np.argmin(d))]
    x = _newton(point)
    return 1.0 if np.isnan(x) or x <= 0 else abs(x)


def _normalize(front):
    m, n = front.shape
    w = np.zeros((n, n)) + 1e-6 + np.eye(n)
    extreme = np.array([int(np.argmin(np.max(front / w[i], axis=1))) for i in range(n)])
    try:
        with np.errstate(all="ignore"):
            a = 1.0 / np.linalg.solve(front[extreme], np.ones(n))
    except np.linalg.LinAlgError:
        a = np.full(n, np.nan)
    if np.any(np.isnan(a)):
        a = front.max(axis=0)
    with np.errstate(all="ignore"):
        front = front / a
    return front - front.min(axis=0), extreme


def _mink_sum(D, k):
    """Row sums of the ``k`` smallest entries (NaN placed last, as missing values)."""
    k = min(k, D.shape[1])
    S = np.sort(np.where(np.isnan(D), np.inf, D), axis=1)[:, :k]
    nan_taken = np.sort(np.isnan(D).astype(int), axis=1)[:, :k].any(axis=1)
    out = S.sum(axis=1)
    out[nan_taken] = np.nan
    return out


def survival_score(front):
    """Crowding of the first front (geodesic distances on the fitted L_p manifold) and the curvature ``p``."""
    m, n = front.shape
    crowd = np.zeros(m)
    selected = np.zeros(m, bool)
    front, extreme = _normalize(front - front.min(axis=0))
    crowd[extreme] = np.inf
    selected[extreme] = True
    p = _geometry(front)
    nn = _lp(front, p)
    with np.errstate(all="ignore"):
        proj = front / _lp(front, p)[:, None]
        mid = 0.5 * proj[:, None, :] + 0.5 * proj[None, :, :]
        pm = mid / _lp(mid, p)[..., None]
        dist = np.sqrt(np.sum((proj[:, None, :] - pm) ** 2, axis=2)) + np.sqrt(np.sum((proj[None, :, :] - pm) ** 2, axis=2))
        np.fill_diagonal(dist, 0.0)
        dist = dist / nn[:, None]
    remaining = list(np.where(~selected)[0])
    for _ in range(m - int(selected.sum()) - 1):
        s = _mink_sum(dist[np.ix_(remaining, np.where(selected)[0])], 2)
        idx = 0 if np.all(np.isnan(s)) else int(np.nanargmax(s))
        best = remaining.pop(idx)
        selected[best] = True
        crowd[best] = s[idx]
    return crowd, p


def environmental_selection(pop, N):
    F = np.round(objs(pop), 12)
    front_no, max_f = nd_sort(F, cons(pop), N)
    nxt = front_no < max_f
    crowd = np.zeros(len(F))
    f1 = front_no == 1
    crowd[f1], p = survival_score(F[f1])
    zmin = F.min(axis=0)
    with np.errstate(all="ignore"):
        for i in range(2, int(max_f) + 1):
            fi = front_no == i
            crowd[fi] = 1.0 / _lp(F[fi] - zmin, p)
    last = np.where(front_no == max_f)[0]
    c = crowd[last]
    rank = np.lexsort((-np.where(np.isnan(c), 0, c), ~np.isnan(c)))   # descending, NaN first
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], crowd[nxt]


class AGEMOEA2(LoopAlgorithm):
    """Approximation-Guided Evolutionary Multi-Objective Algorithm II (AGE-MOEA-II): the first front is modelled as
    an L_p manifold (p by Newton-Raphson on the point nearest to the origin), its members are ranked by geodesic
    crowding and the later fronts by proximity to the ideal point in the L_p norm."""

    ALGO_FLAGS = {"multi", "many", "real", "integer", "constrained"}
    OBJECTIVE_SCOPE = "many"

    def __init__(self, pop_size: int = 100, sampling=None, **kwargs: Any) -> None:
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)

    def start(self):
        _, self.front_no, self.crowd = environmental_selection(self.pop, self.N)

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.crowd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.front_no, self.crowd = environmental_selection(Population.merge(self.pop, off), self.N)
