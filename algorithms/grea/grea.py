# emopylab 2026
"""GrEA (grid-based evolutionary algorithm).

Reference:
S. Yang, M. Li, X. Liu, and J. Zheng. A grid-based evolutionary algorithm for many-objective
optimization. IEEE Transactions on Evolutionary Computation, 2013, 17(5): 721-736.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'GrEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}

_DIV = [0, 45, 15, 10, 9, 9, 8, 8, 10, 12]


def _grid(F, div):
    fmax, fmin = F.max(axis=0), F.min(axis=0)
    lb = fmin - (fmax - fmin) / 2 / div
    ub = fmax + (fmax - fmin) / 2 / div
    d = (ub - lb) / div
    with np.errstate(all="ignore"):
        loc = np.floor((F - lb) / d)
    loc[np.isnan(loc)] = 0
    return loc, lb, d


def _grid_distance(loc):
    gd = np.sum(np.abs(loc[:, None, :] - loc[None, :, :]), axis=2)
    np.fill_diagonal(gd, np.inf)
    return gd


def _last_selection(F, K, div):
    N, M = F.shape
    loc, lb, d = _grid(F, div)
    GR = loc.sum(axis=1)
    GCD = np.zeros(N)
    with np.errstate(all="ignore"):
        GCPD = np.sqrt(np.sum(((F - (lb + loc * d)) / d) ** 2, axis=1))
    GCPD = np.nan_to_num(GCPD)
    GD = _grid_distance(loc)
    less = np.any(loc[:, None, :] < loc[None, :, :], axis=2)
    more = np.any(loc[:, None, :] > loc[None, :, :], axis=2)
    G = less & ~more                                            # grid dominance: i dominates j
    remain = np.ones(N, bool)
    while remain.sum() > N - K:
        can = np.where(remain)[0]
        temp = np.where(GR[can] == GR[can].min())[0]
        temp2 = np.where(GCD[can[temp]] == GCD[can[temp]].min())[0]
        q = can[temp[temp2[int(np.argmin(GCPD[can[temp[temp2]]]))]]]
        remain[q] = False
        GCD = GCD + np.maximum(M - GD[q], 0)
        Eq = (GD[q] == 0) & remain
        Gq = G[q] & remain
        NGq = remain & ~Gq
        Nq = (GD[q] < M) & remain
        GR = GR.copy()
        GR[Eq] += M + 2
        GR[Gq] += M
        PD = np.zeros(N)
        for p in np.where(Nq & NGq & ~Eq)[0]:
            if PD[p] < M - GD[q, p]:
                PD[p] = M - GD[q, p]
                Gp = G[p] & remain
                for r in np.where(Gp & ~(Gq | Eq))[0]:
                    if PD[r] < PD[p]:
                        PD[r] = PD[p]
        pp = NGq & ~Eq
        GR[pp] += PD[pp]
    return ~remain


def _environmental_selection(pop, N, div):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    nxt[last[_last_selection(F[last], N - int(nxt.sum()), div)]] = True
    return pop[nxt]


def _mating_selection(F, div, rng):
    N, M = F.shape
    loc, _, _ = _grid(F, div)
    gd = np.maximum(M - _grid_distance(loc), 0)
    gcd = gd.sum(axis=1)
    p1, p2 = rng.integers(0, N, size=N), rng.integers(0, N, size=N)
    dom = np.any(F[p1] < F[p2], axis=1).astype(int) - np.any(F[p1] > F[p2], axis=1).astype(int)
    gdom = np.any(loc[p1] < loc[p2], axis=1).astype(int) - np.any(loc[p1] > loc[p2], axis=1).astype(int)
    return np.concatenate([p1[(dom == 1) | (gdom == 1)], p2[(dom == -1) | (gdom == -1)],
                           p1[(dom == 0) & (gdom == 0) & (gcd[p1] <= gcd[p2])], p2[(dom == 0) & (gdom == 0) & (gcd[p1] > gcd[p2])]])


class GrEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, div: int | None = None, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.div_arg = div

    def start(self):
        self.div = int(self.div_arg) if self.div_arg else _DIV[min(self.M, 10) - 1]

    def step(self):
        pool = _mating_selection(objs(self.pop), self.div, self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N, self.div)
