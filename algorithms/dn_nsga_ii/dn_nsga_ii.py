# emopylab 2026
"""DN-NSGA-II (decision space based niching NSGA-II).

Reference:
J. Liang, C. Yue, and B. Qu. Multimodal multi-objective optimization: A preliminary study.
Proceedings of the IEEE Congress on Evolutionary Computation, 2016, 2454-2461.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, ga, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'DNNSGAII': {'integer', 'multi', 'multimodal', 'real'}}


def _environmental_selection(pop, N):
    """Front rank, then decision-space crowding distance inside the last front."""
    F, X = objs(pop), decs(pop)
    front_no, max_f = nd_sort(F, cons(pop), N)
    nxt = front_no < max_f
    cd_obj = crowding(F, front_no)
    cd_dec = crowding(X, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd_dec[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], cd_obj[nxt]


def _tournament_mod(K, n, X, rng, *fitness):
    """Binary tournament whose second contender is the decision-space nearest of K-1 random ones."""
    fit = np.column_stack([np.asarray(f, dtype=float).reshape(-1) for f in fitness])
    order = np.lexsort(tuple(fit[:, j] for j in range(fit.shape[1] - 1, -1, -1)))
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    parents = rng.integers(0, len(fit), size=(K, n))
    dist = np.linalg.norm(X[parents[1:]] - X[parents[0]][None, :, :], axis=2)  # (K-1, n)
    second = parents[1:][np.argmin(dist, axis=0) + 0, np.arange(n)] if K > 1 else parents[0]
    pair = np.vstack([parents[0], second])
    best = np.argmin(rank[pair], axis=0)
    return pair[best, np.arange(n)]


class DNNSGAII(LoopAlgorithm):
    def start(self):
        self.pop, self.front_no, self.crowd = _environmental_selection(self.pop, self.N)

    def step(self):
        half = int(round(self.N / 2))
        pool = _tournament_mod(half, half, decs(self.pop), self.rng, self.front_no, -self.crowd)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.front_no, self.crowd = _environmental_selection(Population.merge(self.pop, off), self.N)
