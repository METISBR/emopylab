# emopylab 2026
"""KnEA (knee point driven evolutionary algorithm).

Reference:
X. Zhang, Y. Tian, and Y. Jin. A knee point-driven evolutionary algorithm for many-objective
optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(6): 761-776.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, pdist2, tournament
from core.population import Population

ALGORITHM_FLAGS = {'KnEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _find_knee_points(F, front_no, max_f, r, t, rate):
    N, M = F.shape
    knee = np.zeros(N, bool)
    dist = np.zeros(N)
    for i in range(1, int(max_f) + 1):
        cur = np.where(front_no == i)[0]
        if len(cur) <= M:
            knee[cur] = True
            continue
        rank = np.argsort(-F[cur], axis=0, kind="stable")
        extreme = [int(rank[0, 0])]
        for j in range(1, M):
            k = 0
            e = int(rank[k, j])
            while e in extreme:
                k += 1
                e = int(rank[k, j])
            extreme.append(e)
        hyper = _solve(F[cur[extreme]], np.ones(M))
        dist[cur] = -(F[cur] @ hyper - 1.0) / np.sqrt(np.sum(hyper ** 2))
        fmax, fmin = F[cur].max(axis=0), F[cur].min(axis=0)
        r[i - 1] = 1.0 if t[i - 1] == -1 else r[i - 1] / np.exp((1 - t[i - 1] / rate) / M)
        R = (fmax - fmin) * r[i - 1]
        rank = np.argsort(-dist[cur], kind="stable")
        choose = np.zeros(len(cur), bool)
        remain = np.ones(len(cur), bool)
        for j in rank:
            if remain[j]:
                near = np.all(np.abs(F[cur[j]] - F[cur]) <= R, axis=1)
                remain[near] = False
                choose[j] = True
        t[i - 1] = choose.sum() / len(cur)
        chosen_in_rank = np.where(choose[rank])[0]
        choose[rank[chosen_in_rank[-1]]] = False
        knee[cur[choose]] = True
    return knee, dist, r, t


def _environmental_selection(pop, front_no, max_f, knee, dist, K):
    nxt = front_no < max_f
    nxt[knee] = True
    if nxt.sum() < K:
        temp = np.where((front_no == max_f) & ~knee)[0]
        rank = np.argsort(-dist[temp], kind="stable")
        nxt[temp[rank[: K - int(nxt.sum())]]] = True
    elif nxt.sum() > K:
        temp = np.where((front_no == max_f) & knee)[0]
        rank = np.argsort(dist[temp], kind="stable")
        nxt[temp[rank[: int(nxt.sum()) - K]]] = False
    return pop[nxt], front_no[nxt], knee[nxt]


class KnEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, rate: float = 0.5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.rate = float(rate)

    def start(self):
        c = cons(self.pop)
        self.front_no, _ = nd_sort(objs(self.pop), c if c.size else None, np.inf)
        self.knee = np.zeros(self.N, bool)
        self.r = -np.ones(2 * self.N)
        self.t = -np.ones(2 * self.N)

    def _mating_selection(self):
        F = objs(self.pop)
        dis = pdist2(F, F)
        np.fill_diagonal(dis, np.inf)
        dis = np.sort(dis, axis=0)
        crowd = np.sum(dis[:3] * np.array([[3.0], [2.0], [1.0]]), axis=0)
        return tournament(2, len(F), self.front_no, -self.knee.astype(float), -crowd, rng=self.rng)

    def step(self):
        pool = self._mating_selection()
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        pop = Population.merge(self.pop, off)
        c = cons(pop)
        front_no, max_f = nd_sort(objs(pop), c if c.size else None, self.N)
        knee, dist, self.r, self.t = _find_knee_points(objs(pop), front_no, max_f, self.r, self.t, self.rate)
        self.pop, self.front_no, self.knee = _environmental_selection(pop, front_no, max_f, knee, dist, self.N)
