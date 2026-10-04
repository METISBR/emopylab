# emopylab 2026
"""MaOEA-DDFC (many-objective evolutionary algorithm based on directional diversity and).

Reference:
J. Cheng, G. G. Yen, and G. Zhang. A many-objective evolutionary algorithm with enhanced mating and
environmental selections. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 592-605.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, pdist2, roulette
from core.population import Population

ALGORITHM_FLAGS = {'MaOEADDFC': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _cal_fc(F, zmin):
    N, M = F.shape
    G = F - zmin
    w = np.zeros((N, M))
    bound = np.any(G == 0, axis=1)                      # objective value equal to the ideal point
    w[bound[:, None] & (G == 0)] = 1.0
    nb = ~bound
    with np.errstate(all="ignore"):
        inv = 1.0 / G[nb]
        w[nb] = inv / inv.sum(axis=1, keepdims=True)
    return np.maximum(np.max(w * G, axis=1), 1e-6)


def _last_selection(P, F, zmin, total, K, L, rng):
    A = np.vstack([P, F])
    N, M = A.shape
    fc = _cal_fc(A, zmin)
    zmin = A.min(axis=0)
    W = np.full((M, M), 1e-6)
    np.fill_diagonal(W, 1.0)
    asf = np.stack([np.max((A - zmin) / W[i], axis=1) for i in range(M)], axis=1)
    extreme = np.argmin(asf, axis=0)
    try:
        hyper = np.linalg.solve(A[extreme], np.ones(M))
        with np.errstate(all="ignore"):
            a = 1.0 / hyper
    except np.linalg.LinAlgError:
        a = np.full(M, np.nan)
    if np.any(np.isnan(a)):
        a = A.max(axis=0)
    with np.errstate(all="ignore"):
        A = (A - zmin) / (a - zmin)
        A = A / A.sum(axis=1, keepdims=True)
    A = np.nan_to_num(A)
    choose = np.zeros(N, bool)
    choose[: len(P)] = True
    dist = pdist2(A, A)
    np.fill_diagonal(dist, np.inf)
    while choose.sum() < total:
        src = choose if choose.any() else ~choose
        dis = np.sort(dist[np.ix_(~choose, src)], axis=1)
        with np.errstate(all="ignore"):
            dd = np.sum(1.0 / dis[:, : min(K, dis.shape[1])], axis=1)
        remain = np.where(~choose)[0]
        R = remain[np.argsort(dd, kind="stable")[: min(L, len(remain))]]
        choose[R[int(roulette(1, fc[R], rng=rng)[0])]] = True
    return choose[len(P):]


def _environmental_selection(pop, zmin, N, K, L, rng):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    choose = _last_selection(F[nxt], F[last], zmin, N, K, L, rng)
    nxt[last[choose]] = True
    return pop[nxt]


class MaOEADDFC(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, K: int = 5, L: int = 3, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.K, self.L = int(K), int(L)

    def start(self):
        self.zmin = objs(self.pop).min(axis=0)

    def step(self):
        rng, N = self.rng, len(self.pop)
        F = objs(self.pop)
        fc = _cal_fc(F, self.zmin)
        p1, p2 = rng.integers(0, N, size=N), rng.integers(0, N, size=N)
        dom = np.any(F[p1] < F[p2], axis=1).astype(int) - np.any(F[p1] > F[p2], axis=1).astype(int)
        pool = np.concatenate([p1[dom == 1], p2[dom == -1], p1[(dom == 0) & (fc[p1] <= fc[p2])], p2[(dom == 0) & (fc[p1] > fc[p2])]])
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=rng))
        self.zmin = np.minimum(self.zmin, objs(off).min(axis=0))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.zmin, self.N, self.K, self.L, rng)
