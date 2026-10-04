# emopylab 2026
"""RPEA (reference points-based evolutionary algorithm).

Reference:
Y. Liu, D. Gong, X. Sun, and Y. Zhang. Many-objective evolutionary optimization based on reference
points. Applied Soft Computing, 2017, 50: 344-355.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, first_front, ga, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'RPEA': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def _tchebychev_distance(F, R):
    with np.errstate(all="ignore"):
        return np.max((F[:, None, :] - R[None, :, :]) / (F.max(axis=0) - F.min(axis=0)) / F.shape[1], axis=2)


def _crowding_each_obj(F):
    N, M = F.shape
    cd = np.zeros((N, M))
    span = F.max(axis=0) - F.min(axis=0)
    for i in range(M):
        rank = np.argsort(F[:, i], kind="stable")
        cd[rank[0], i] = np.inf
        cd[rank[-1], i] = np.inf
        if N > 2:
            with np.errstate(all="ignore"):
                cd[rank[1:-1], i] = (F[rank[2:], i] - F[rank[:-2], i]) / span[i]
    return cd


def _generate_ref_points(Q, diff, alpha, N):
    F = objs(Q[first_front(objs(Q))])
    subN = min(len(F), int(np.ceil(alpha * N)))
    rank = np.argsort(-_crowding_each_obj(F), axis=0, kind="stable")
    R = []
    for m in range(F.shape[1]):
        Rm = F[rank[:subN, m]].copy()
        Rm[:, m] -= diff[m]
        R.append(Rm)
    R = np.vstack(R)
    R = R[first_front(R)]
    if len(R) > N:
        order = np.argsort(-_crowding_each_obj(R).sum(axis=1), kind="stable")
        R = R[order[:N]]
    return R


def _environmental_selection(pop, R, N):
    dist = _tchebychev_distance(objs(pop), R)
    remain_p = list(range(len(pop)))
    remain_r = list(range(len(R)))
    target = len(pop) - N
    while len(remain_p) > target:
        if not remain_r:
            remain_r = list(range(len(R)))
        sub = dist[np.ix_(remain_p, remain_r)]
        imin = np.argmin(sub, axis=0)
        temp = sub[imin, np.arange(sub.shape[1])]
        jmin = int(np.argmin(temp))
        del remain_p[int(imin[jmin])]
        del remain_r[jmin]
    return pop[np.setdiff1d(np.arange(len(pop)), remain_p)]


class RPEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, alpha: float = 0.4, delta: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha, self.delta = float(alpha), float(delta)

    def _diff(self):
        F = objs(self.pop)
        return self.delta * (F.max(axis=0) - F.min(axis=0))

    def start(self):
        self.R = _generate_ref_points(self.pop, self._diff(), self.alpha, self.N)

    def step(self):
        fit = _tchebychev_distance(objs(self.pop), self.R).min(axis=1)
        pool = tournament(2, self.N, fit, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        merged = Population.merge(self.pop, off)
        self.R = _generate_ref_points(merged, self._diff(), self.alpha, self.N)
        self.pop = _environmental_selection(merged, self.R, self.N)
