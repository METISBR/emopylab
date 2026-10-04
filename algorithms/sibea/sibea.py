# emopylab 2026
"""SIBEA (simple indicator-based evolutionary algorithm).

Reference:
E. Zitzler, D. Brockhoff, and L. Thiele. The hypervolume indicator revisited: On the design of
Pareto-compliant indicators via weighted integration. Proceedings of the International Conference on
Evolutionary Multi-Criterion Optimization, 2007, 862-876.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs
from core.population import Population
from util.hv import hv_contributions, hypervolume

ALGORITHM_FLAGS = {'SIBEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def hv_loss(F, front_no):
    """Hypervolume lost by removing each solution from its own front (reference point = max + 0.1)."""
    loss = np.zeros(len(F))
    ref = F.max(axis=0) + 0.1
    for f in np.unique(front_no[np.isfinite(front_no)]):
        cur = np.where(front_no == f)[0]
        loss[cur] = hv_contributions(F[cur], ref)
    return loss


def _environmental_selection(pop, N):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    loss = hv_loss(F[last], front_no[last])
    rank = np.argsort(-loss, kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt]


class SIBEA(LoopAlgorithm):
    def step(self):
        pool = self.rng.integers(0, len(self.pop), size=self.N)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N)
