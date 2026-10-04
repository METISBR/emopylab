# emopylab 2026
"""MOEA-IGD-NS (multi-objective evolutionary algorithm based on an enhanced IGD).

Reference:
Y. Tian, X. Zhang, R. Cheng, and Y. Jin. A multi-objective evolutionary algorithm based on an
enhanced inverted generational distance metric. Proceedings of the IEEE Congress on Evolutionary
Computation, 2016, 5222-5229.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, angle_matrix, decs, first_front, ga, nd_sort, objs, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'MOEAIGDNS': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _update_archive(pop, NA, rng):
    pop = pop[first_front(objs(pop))]
    F = objs(pop)
    choose = np.zeros(len(pop), bool)
    choose[np.argmax(F, axis=0)] = True
    if choose.sum() > NA:
        sel = np.where(choose)[0]
        choose = np.zeros(len(pop), bool)
        choose[sel[rng.permutation(len(sel))[:NA]]] = True
    else:
        cosine = np.cos(angle_matrix(F))
        np.fill_diagonal(cosine, 0.0)
        while choose.sum() < NA and not choose.all():
            un = np.where(~choose)[0]
            x = np.argmin(cosine[np.ix_(~choose, choose)].max(axis=1))
            choose[un[x]] = True
    return pop[choose]


def _last_selection(F, W, K):
    """Keep K of the rows of F minimising the IGD-NS increase; returns a boolean mask."""
    N, NW = len(F), len(W)
    dist = pdist2(F, W)
    con = dist.min(axis=1)
    rank = np.argsort(dist, axis=0, kind="stable")
    dis = np.take_along_axis(dist, rank, axis=0)
    remain = np.ones(N, bool)
    while remain.sum() > K:
        outliers = remain.copy()
        outliers[rank[0]] = False
        metric = dis[0].sum() + con[outliers].sum()
        metrics = np.full(N, np.inf)
        metrics[outliers] = metric - con[outliers]
        for p in np.where(remain & ~outliers)[0]:
            temp = rank[0] == p
            out = np.zeros(N, bool)
            out[rank[1, temp]] = True
            out &= outliers
            metrics[p] = metric - dis[0, temp].sum() + dis[1, temp].sum() - con[out].sum()
        dele = int(np.argmin(metrics))
        keep = rank != dele
        L = int(remain.sum())
        dis = dis.T[keep.T].reshape(NW, L - 1).T
        rank = rank.T[keep.T].reshape(NW, L - 1).T
        remain[dele] = False
    return remain


def _environmental_selection(pop, W, N):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    nxt[last[_last_selection(F[last], W, N - int(nxt.sum()))]] = True
    return pop[nxt]


class MOEAIGDNS(LoopAlgorithm):
    def start(self):
        self.archive = _update_archive(self.pop, 5 * self.N, self.rng)

    def step(self):
        pool = self.rng.integers(0, len(self.pop), size=self.N)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.archive = _update_archive(Population.merge(self.archive, off), 5 * self.N, self.rng)
        self.pop = _environmental_selection(Population.merge(self.pop, off), objs(self.archive), self.N)
