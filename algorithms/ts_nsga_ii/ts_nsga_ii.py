# emopylab 2026
"""TS-NSGA-II (two stage NSGA-II).

Reference:
F. Ming, W. Gong, and L. Wang. A two-stage evolutionary algorithm with balanced convergence and
diversity for many-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics:
Systems, 2022, 52(10): 6222-6234.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, ga, nd_sort, objs, tournament, uniform_point
from algorithms.rpd_nsga_ii.rpd_nsga_ii import _environmental_selection as _spd_selection
from core.population import Population

ALGORITHM_FLAGS = {'TSNSGAII': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _level_sort(F, n):
    zmin, zmax = F.min(axis=0), F.max(axis=0)
    interval = (zmax - zmin) / n
    with np.errstate(all="ignore"):
        r = np.nan_to_num((F - zmin) / interval)
    return np.maximum(1, np.ceil(r.max(axis=1)).astype(int) - 1)


def _density(pop, W):
    F = objs(pop)
    with np.errstate(all="ignore"):
        Fn = np.nan_to_num((F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0)))
    region = np.argmin(cosine_distance(Fn, W), axis=1)
    counts = np.bincount(region, minlength=len(W))
    return counts[region].astype(float)


def _environmental_selection_1(pop, W, N, M):
    n = 2 * M
    F = objs(pop)
    front_no, max_n = nd_sort(F, None, np.inf)
    keep = []
    for i in range(1, int(max_n) + 1):
        keep += list(np.where(front_no == i)[0])
        if len(keep) >= N:
            break
    nds = pop[np.asarray(keep)]
    level = _level_sort(objs(nds), n)
    nds = nds[np.isin(level, np.arange(1, n + 1))]        # the level loop of the reference never breaks early
    chosen = []
    remaining = list(range(len(nds)))
    for i in range(len(W)):
        if not remaining:
            break
        Fr = objs(nds[np.asarray(remaining)])
        with np.errstate(all="ignore"):
            Fn = np.nan_to_num((Fr - Fr.min(axis=0)) / (Fr.max(axis=0) - Fr.min(axis=0)))
        k = int(np.argmin(cosine_distance(Fn, W[i:i + 1])[:, 0]))
        chosen.append(remaining.pop(k))
    return nds[np.asarray(chosen)]


class TSNSGAII(LoopAlgorithm):
    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        self.pop, self.front_no, self.d2 = _spd_selection(self.pop, self.W, self.N)

    def step(self):
        pool = tournament(2, self.N, self.front_no, self.d2, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        merged = Population.merge(self.pop, off)
        if self.FE < (0.8 + 0.2 * (1 - 3 / self.M)) * self.max_FE:
            self.pop, self.front_no, self.d2 = _spd_selection(merged, self.W, self.N)
        else:
            self.pop = _environmental_selection_1(merged, self.W, self.N, self.M)
            self.front_no, _ = nd_sort(objs(self.pop), None, np.inf)
            self.d2 = _density(self.pop, self.W)
