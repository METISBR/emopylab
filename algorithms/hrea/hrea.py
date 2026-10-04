# emopylab 2026
"""HREA (hierarchy ranking based evolutionary algorithm).

Reference:
W. Li, X. Yao, T. Zhang, R. Wang, and L. Wang. Hierarchy ranking method for multimodal multi-
objective optimization with local Pareto fronts. IEEE Transactions on Evolutionary Computation,
2023, 27(1): 98-110.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, pdist2, tournament
from core.population import Population

ALGORITHM_FLAGS = {'HREA': {'integer', 'multi', 'multimodal', 'real'}}


def _crowding_dec(X):
    N = len(X)
    if N < 2:
        return np.full(N, np.inf)
    Z, Zmax = X.min(axis=0), X.max(axis=0)
    with np.errstate(all="ignore"):
        P = (X - Z) / (Zmax - Z)
    d = np.sort(np.sqrt(np.maximum(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2), 0)), axis=1)
    with np.errstate(all="ignore"):
        return (N - 1) / np.sum(1.0 / d[:, 1:], axis=1)


def _range_v(X):
    return 0.2 * np.prod(X.max(axis=0) - X.min(axis=0)) ** (1.0 / X.shape[1])


def environmental_selection(pop, N):
    n = len(pop)
    X, F = decs(pop), objs(pop)
    dist = pdist2(X, X)
    V = _range_v(X)
    dom = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] > V:
                continue
            L1, L2 = F[i] < F[j], F[i] > F[j]
            if np.all(L1 | ~L2):
                dom[j, i] = 1
            elif np.all(L2 | ~L1):
                dom[i, j] = 1
    near = dist < V
    local = np.array([dom[i, near[i]].sum() / near[i].sum() for i in range(n)])
    crowd = np.sort(dist, axis=0)[:3].sum(axis=0)
    order = np.lexsort((-crowd, local))
    pop = pop[order]
    if len(pop) > N:
        pop = pop[:N]
    return pop, _crowding_dec(decs(pop))


def archive_update(pop, N, eps, st):
    n = len(pop)
    if eps != 1 and st < 0.5:
        eps = 2 * (1 - eps) / (2 * st + 1) + 2 * eps - 1
    front, _ = nd_sort(objs(pop), None, n)
    nxt = front == 1
    first = pop[nxt]
    new = first
    remain = pop[~nxt]
    V = _range_v(decs(pop))
    while len(remain) > 0:
        d = pdist2(decs(new), decs(remain)).min(axis=0)
        remain = remain[~(d < V)]
        if len(remain) == 0:
            break
        f2, _ = nd_sort(objs(remain), None, len(remain))
        pick = remain[f2 == 1]
        nF, _ = nd_sort(np.vstack([objs(pick) * (1 - eps), objs(first)]), None, len(pick) + len(first))
        nF = nF[: len(pick)]
        if nF.max() > 1:
            new = Population.merge(new, pick[nF == 1])
            remain = remain[f2 != 1]
            break
        new = Population.merge(new, pick)
        remain = remain[f2 != 1]
    pop = new
    if len(pop) > N:
        awd = []
        front, maxf = nd_sort(objs(pop), None, len(pop))
        n_sub = int(np.ceil(N / maxf))
        sel, tmp = [], []
        for i in range(1, int(maxf) + 1):
            p = pop[front == i]
            if len(p) < n_sub:
                sel.append(p)
                awd += [n_sub - len(p)] * len(p)
            else:
                tmp.append(p)
        sel = Population.merge(*sel) if sel else None
        tmp = Population.merge(*tmp) if tmp else None
        n_sel = 0 if sel is None else len(sel)
        while tmp is not None and len(tmp) > N - n_sel:
            d = np.sort(pdist2(decs(tmp), decs(tmp)), axis=0)
            s = d[:3].sum(axis=0)
            tmp = tmp[np.delete(np.arange(len(tmp)), int(np.argmin(s)))]
        n_tmp = 0 if tmp is None else len(tmp)
        awd = np.array(awd + [0] * n_tmp, dtype=float) + 1
        pop = Population.merge(*[p for p in (sel, tmp) if p is not None])
        return pop, _crowding_dec(decs(pop)) * awd
    return pop, _crowding_dec(decs(pop))


class HREA(LoopAlgorithm):
    """Hierarchical-ranking multimodal EA: a decision-space niche-aware ranking population and an e-dominance
    archive (whose tolerance relaxes and tightens along the run) provide mating pools for a GA."""

    def __init__(self, pop_size: int = 100, eps: float = 0.3, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.eps, self.p = float(eps), 0.5

    def start(self):
        self.P = self.pop
        _, self.cd1 = environmental_selection(self.pop, self.N)
        self.archive, self.cd2 = archive_update(self.pop, self.N, self.eps, 0.0)
        self.pop = self.archive

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        if self.FE >= self.max_FE * 0.5 and rng.random() < self.p:
            pool = tournament(2, N, -self.cd2, rng=rng)
            off = self.evaluate(ga(pr, decs(self.archive[pool]), rng=rng))
        else:
            pool = tournament(2, N, -self.cd1, rng=rng)
            off = self.evaluate(ga(pr, decs(self.P[pool]), rng=rng))
        self.P, self.cd1 = environmental_selection(Population.merge(self.P, off), N)
        self.archive, self.cd2 = archive_update(Population.merge(self.archive, off), N, self.eps, self.FE / self.max_FE)
        self.pop = self.archive
