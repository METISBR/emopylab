# emopylab 2026
"""I-SIBEA (interactive simple indicator-based evolutionary algorithm).

Reference:
T. Chugh, K. Sindhya, J. Hakanen, and K. Miettinen. An interactive simple indicator-based
evolutionary algorithm (I-SIBEA) for multiobjective optimization problems. Proceedings of the
International Conference on Evolutionary Multi-Criterion Optimization, 2015, 277-291.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, first_front, ga, nd_sort, objs, tournament
from core.population import Population
from util.hv_slice import cal_whv

ALGORITHM_FLAGS = {'ISIBEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _evaluate(F, point):
    flag = np.all(F <= point, axis=1) | np.all(F >= point, axis=1)
    F = F.copy()
    F[~flag] += 1e10
    return F


def _whv_loss(F, front_no, wz=None, AA=None, RA=None):
    n = len(F)
    weight = np.ones(n)
    if wz is not None:
        for i in range(n):
            if np.any(np.all(F[i] <= AA, axis=1)):
                weight[i] = wz[2]
            elif np.any(np.all(F[i] > RA, axis=1)):
                weight[i] = wz[0]
            else:
                weight[i] = wz[1]
    loss = np.zeros(n)
    ref = F.max(axis=0) + 0.1
    for f in np.unique(front_no[np.isfinite(front_no)]):
        cur = np.where(front_no == f)[0]
        total = cal_whv(F[cur], ref, weight[cur])
        for i in range(len(cur)):
            rest = np.delete(cur, i)
            loss[cur[i]] = total - cal_whv(F[rest], ref, weight[rest])
    return loss


def _interaction(F, point):
    F = np.unique(F, axis=0)
    F = F[first_front(F)]
    ideal = F.min(axis=0)
    fit = np.max((F - ideal) / point, axis=1)
    pref = int(np.argmin(fit))
    AA, RA = F[pref:pref + 1], np.delete(F, pref, axis=0)
    ref = F.max(axis=0) + 0.1
    ratio = cal_whv(AA, ref, np.ones(len(AA))) / max(cal_whv(RA, ref, np.ones(len(RA))), 1e-300) if len(RA) else 1.0
    return np.array([0.0, 1.0, 1.0 + ratio]), AA, RA


def _environmental_selection(pop, N, wz, AA, RA):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    loss = _whv_loss(F, front_no, wz, AA, RA)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-loss[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], loss[nxt]


class ISIBEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, Point=None, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Point = None if Point is None else np.asarray(Point, dtype=float)

    def start(self):
        self.point = np.ones(self.M) if self.Point is None else self.Point
        F = objs(self.pop)
        self.front_no, _ = nd_sort(_evaluate(F, self.point), None, np.inf)
        self.loss = _whv_loss(F, self.front_no)
        self.wz = self.AA = self.RA = None

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.loss, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.front_no, self.loss = _environmental_selection(Population.merge(self.pop, off), self.N, self.wz, self.AA, self.RA)
        if int(np.ceil(self.FE / self.N)) % int(np.ceil(np.ceil(self.max_FE / self.N) / 4)) == 0:
            self.wz, self.AA, self.RA = _interaction(objs(self.pop), self.point)
