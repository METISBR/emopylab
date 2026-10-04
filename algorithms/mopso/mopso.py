# emopylab 2026
"""MOPSO (multi-objective particle swarm optimization).

Reference:
C. A. Coello Coello and M. S. Lechuga. MOPSO: A proposal for multiple objective particle swarm
optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2002, 1051-1056.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, first_front, grid_locations, objs, roulette
from core.population import Population

ALGORITHM_FLAGS = {'MOPSO': {'integer', 'multi', 'real'}}


def _rep_selection(F, N, div, rng):
    _, site, crowd_g = grid_locations(F, div)
    the_grid = roulette(N, crowd_g.astype(float), rng=rng)
    rep = np.zeros(N, dtype=int)
    for i in range(N):
        in_grid = np.where(site == the_grid[i])[0]
        rep[i] = in_grid[rng.integers(0, len(in_grid))]
    return rep


def _delete(F, K, div, rng):
    n = len(F)
    _, site, crowd_g = grid_locations(F, div)
    site, crowd_g = site.astype(float), crowd_g.astype(float)
    dele = np.zeros(n, bool)
    while dele.sum() < K:
        mg = np.where(crowd_g == crowd_g.max())[0]
        grid = mg[rng.integers(0, len(mg))]
        in_grid = np.where(site == grid)[0]
        p = in_grid[rng.integers(0, len(in_grid))]
        dele[p] = True
        site[p] = np.nan
        crowd_g[grid] -= 1
    return dele


def _update_archive(A, N, div, rng):
    A = A[first_front(objs(A))]
    if len(A) > N:
        A = A[~_delete(objs(A), len(A) - N, div, rng)]
    return A


def _update_pbest(pbest, pop, rng):
    temp = objs(pbest) - objs(pop)
    dom = np.any(temp < 0, axis=1).astype(int) - np.any(temp > 0, axis=1).astype(int)
    out = pbest.copy(deep=False)
    rep = (dom == -1) | ((dom == 0) & (rng.random(len(dom)) < 0.5))
    out[rep] = pop[rep]
    return out


class MOPSO(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, div: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.div = int(div)

    def start(self):
        self.swarm = self.pop
        self.archive = _update_archive(self.pop, self.N, self.div, self.rng)
        self.pbest = self.pop
        self.pop = self.archive

    def step(self):
        rep = _rep_selection(objs(self.archive), self.N, self.div, self.rng)
        self.swarm = self.pso(self.swarm, self.pbest, self.archive[rep])
        self.archive = _update_archive(Population.merge(self.archive, self.swarm), self.N, self.div, self.rng)
        self.pbest = _update_pbest(self.pbest, self.swarm, self.rng)
        self.pop = self.archive
