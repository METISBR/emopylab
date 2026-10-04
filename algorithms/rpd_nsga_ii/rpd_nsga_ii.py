# emopylab 2026
"""RPD-NSGA-II (reference point dominance-based NSGA-II).

Reference:
M. Elarbi, S. Bechikh, A. Gupta, L. B. Said, and Y. S. Ong. A new decomposition-based NSGA-II for
many-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2018,
48(7): 1191-1210.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, first_front, ga, nd_sort, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'RPDNSGAII': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _nrpdd_sort(F, d1, d2, RP, n_sort):
    """Pareto sorting where mutually non-dominated members of one reference-point niche are ordered by d1+5*d2."""
    N = len(F)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    rel = np.where(le & lt, 1, np.where(le.T & lt.T, -1, 0))
    tie = (rel == 0) & (RP[:, None] == RP[None, :])
    pen = d1 + 5 * d2
    rel = np.where(tie & (pen[:, None] < pen[None, :]), 1, np.where(tie & (pen[:, None] > pen[None, :]), -1, rel))
    front = np.full(N, np.inf)
    max_f = 0
    while np.sum(np.isfinite(front)) < min(n_sort, N):
        max_f += 1
        D = np.isfinite(front)
        for i in range(N):
            if D[i]:
                continue
            cand = np.where(~D[i + 1:])[0] + i + 1
            r = rel[i, cand]
            worse = np.where(r == -1)[0]
            if len(worse):
                D[cand[: worse[0]][r[: worse[0]] == 1]] = True
                D[i] = True
            else:
                D[cand[r == 1]] = True
                front[i] = max_f
    return front, max_f


def _environmental_selection(pop, RP_set, N):
    F = objs(pop)
    span = F.max(axis=0) - F.min(axis=0)
    with np.errstate(all="ignore"):
        Fn = (F - F.min(axis=0)) / span
    Fn = np.nan_to_num(Fn)
    norm_p = np.linalg.norm(Fn, axis=1)
    cosine = 1.0 - cosine_distance(Fn, RP_set)
    d1m = norm_p[:, None] * cosine
    d2m = norm_p[:, None] * np.sqrt(np.maximum(0.0, 1 - cosine ** 2))
    RP = np.argmin(d2m, axis=1)
    d2 = d2m[np.arange(len(F)), RP]
    d1 = d1m[np.arange(len(F)), RP]
    nd = np.where(first_front(Fn))[0]
    ext = nd[np.argmax(Fn[nd], axis=0)]
    d1[ext] = 0
    d2[ext] = 0
    front_no, max_f = _nrpdd_sort(Fn, d1, d2, RP, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(d2[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], d2[nxt]


class RPDNSGAII(LoopAlgorithm):
    def initial_size(self):
        self.RP, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        self.pop, self.front_no, self.d2 = _environmental_selection(self.pop, self.RP, self.N)

    def step(self):
        pool = tournament(2, self.N, self.front_no, self.d2, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.front_no, self.d2 = _environmental_selection(Population.merge(self.pop, off), self.RP, self.N)
