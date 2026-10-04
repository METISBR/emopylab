# emopylab 2026
"""NSGA-II-SDR (nSGA-II with strengthened dominance relation).

Reference:
Y. Tian, R. Cheng, X. Zhang, Y. Su, and Y. Jin. A strengthened dominance relation considering
convergence and diversity for evolutionary many- objective optimization. IEEE Transactions on
Evolutionary Computation, 2019, 23(2): 331-345.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, angle_matrix, crowding, decs, ga, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'NSGAIISDR': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def _nd_sort_sdr(F, n_sort):
    N = len(F)
    norm_p = F.sum(axis=1)
    angle = angle_matrix(F)
    np.fill_diagonal(angle, np.pi / 2)                      # cosine of a point with itself is set to 0
    temp = np.unique(angle_min := np.min(np.where(np.eye(N, dtype=bool), np.inf, angle), axis=1)) if N > 1 else np.array([1.0])
    min_a = temp[min(int(np.ceil(N / 2)), len(temp)) - 1]
    theta = np.maximum(1.0, angle / min_a) if min_a > 0 else np.ones_like(angle)
    dominate = norm_p[:, None] * theta < norm_p[None, :]
    np.fill_diagonal(dominate, False)
    front_no = np.full(N, np.inf)
    max_f = 0
    while np.sum(front_no != np.inf) < min(n_sort, N):
        max_f += 1
        current = ~dominate.any(axis=0) & (front_no == np.inf)
        if not current.any():                               # numerical safety: rank everything left
            current = front_no == np.inf
        front_no[current] = max_f
        dominate[current, :] = False
    return front_no, max_f


def _environmental_selection(pop, N, zmin, zmax):
    F = objs(pop) - zmin
    rng_ = zmax - zmin
    if 0.05 * rng_.max() < rng_.min():
        F = F / rng_
    _, x = np.unique(np.round(F * 1e6) / 1e6, axis=0, return_index=True)
    F, pop = F[x], pop[x]
    N = min(N, len(pop))
    front_no, max_f = _nd_sort_sdr(F, N)
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], cd[nxt]


class NSGAIISDR(LoopAlgorithm):
    def start(self):
        F = objs(self.pop)
        self.zmin, self.zmax = F.min(axis=0), F.max(axis=0)
        self.pop, self.front_no, self.crowd = _environmental_selection(self.pop, self.N, self.zmin, self.zmax)

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.crowd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.zmin = np.minimum(self.zmin, objs(off).min(axis=0))
        self.zmax = objs(self.pop[self.front_no == 1]).max(axis=0)
        self.pop, self.front_no, self.crowd = _environmental_selection(Population.merge(self.pop, off), self.N, self.zmin, self.zmax)
