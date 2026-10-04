# emopylab 2026
"""MOMBI-II (many objective metaheuristic based on the R2 indicator II).

Reference:
R. Hernandez Gomez and C. A. Coello Coello. Improved metaheuristic based on the R2 indicator for
many-objective optimization. Proceedings of the Annual Conference on Genetic and Evolutionary
Computation, 2015, 679-686.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MOMBIII': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _r2_ranking(F, W, zmin, zmax):
    N = len(F)
    with np.errstate(all="ignore"):
        G = (F - zmin) / (zmax - zmin)
    norm = np.linalg.norm(G, axis=1)
    with np.errstate(all="ignore"):
        asf = np.max(G[:, None, :] / W[None, :, :], axis=2)                       # (N, NW)
    best = np.full(N, N + 1)
    for i in range(len(W)):
        order = np.lexsort((norm, asf[:, i]))
        pos = np.empty(N, dtype=int)
        pos[order] = np.arange(1, N + 1)
        best = np.minimum(best, pos)
    return best.astype(float), norm


def _update_reference_points(F, zmin, zmax, record, mark, alpha, epsilon):
    z, znad = F.min(axis=0), F.max(axis=0)
    zmin = np.minimum(zmin, z)
    record = np.vstack([record[1:], znad])
    v = record[-2] - znad
    new_mark = np.zeros(len(zmax), bool)
    zmax = zmax.copy()
    if v.max() > alpha:
        zmax = znad
    else:
        for i in range(len(zmax)):
            if abs(zmax[i] - zmin[i]) < epsilon:
                zmax[i] = zmax.max()
                new_mark[i] = True
            elif znad[i] > zmax[i]:
                zmax[i] = 2 * znad[i] - zmax[i]
                new_mark[i] = True
            elif v[i] == 0 and not mark[:, i].any():
                zmax[i] = (zmax[i] + record[:, i].max()) / 2
                new_mark[i] = True
    return zmin, zmax, record, np.vstack([mark[1:], new_mark])


class MOMBIII(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, alpha: float = 0.5, epsilon: float = 0.001, recordSize: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha, self.epsilon, self.record_size = float(alpha), float(epsilon), int(recordSize)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        self.zmin, self.zmax = F.min(axis=0), F.max(axis=0)
        self.record = np.tile(self.zmax, (self.record_size, 1))
        self.mark = np.zeros((self.record_size, self.M), bool)
        self.rank, self.norm = _r2_ranking(F, self.W, self.zmin, self.zmax)

    def step(self):
        pool = tournament(2, self.N, self.rank, self.norm, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        merged = Population.merge(self.pop, off)
        rank, norm = _r2_ranking(objs(merged), self.W, self.zmin, self.zmax)
        keep = np.lexsort((norm, rank))[: self.N]
        self.pop, self.rank, self.norm = merged[keep], rank[keep], norm[keep]
        self.zmin, self.zmax, self.record, self.mark = _update_reference_points(
            objs(self.pop), self.zmin, self.zmax, self.record, self.mark, self.alpha, self.epsilon)
