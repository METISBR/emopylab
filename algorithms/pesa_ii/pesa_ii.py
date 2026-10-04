# emopylab 2026
"""PESA-II (pareto envelope-based selection algorithm II).

Reference:
D. W. Corne, N. R. Jerram, J. D. Knowles, and M. J. Oates. PESA-II: Region-based selection in
evolutionary multiobjective optimization. Proceedings of the Annual Conference on Genetic and
Evolutionary Computation, 2001, 283-290.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, first_front, ga, objs
from core.population import Population

ALGORITHM_FLAGS = {'PESAII': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _grid(F, div):
    d = (F.max(axis=0) - F.min(axis=0)) / div
    with np.errstate(all="ignore"):
        loc = np.floor((F - F.min(axis=0)) / d)
    loc[loc >= div] = div - 1
    loc[np.isnan(loc)] = 0
    uniq, site = np.unique(loc, axis=0, return_inverse=True)
    return uniq, site.reshape(-1), np.bincount(site.reshape(-1), minlength=len(uniq))


def _mating_selection(F, N, div, rng):
    uniq, site, crowd_g = _grid(F, div)
    pool = np.zeros(N, int)
    for i in range(N):
        g = rng.integers(0, len(uniq), size=2)
        best = g[int(np.argmin(crowd_g[g]))]
        cur = np.where(site == best)[0]
        pool[i] = cur[rng.integers(0, len(cur))]
    return pool


def _delete(F, K, div, rng):
    N = len(F)
    _, site, crowd_g = _grid(F, div)
    site = site.astype(float)
    crowd_g = crowd_g.astype(float)
    dele = np.zeros(N, bool)
    while dele.sum() < K:
        max_grid = np.where(crowd_g == crowd_g.max())[0]
        grid = max_grid[rng.integers(0, len(max_grid))]
        in_grid = np.where(site == grid)[0]
        p = in_grid[rng.integers(0, len(in_grid))]
        dele[p] = True
        site[p] = np.nan
        crowd_g[grid] -= 1
    return dele


def _environmental_selection(pop, N, div, rng):
    nxt = first_front(objs(pop))
    if nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_delete(objs(pop)[nxt], int(nxt.sum()) - N, div, rng)]] = False
    return pop[nxt]


class PESAII(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, div: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.div = int(div)

    def step(self):
        pool = _mating_selection(objs(self.pop), self.N, self.div, self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N, self.div, self.rng)
