# emopylab 2026
"""BiGE (bi-goal evolution).

Reference:
M. Li, S. Yang, and X. Liu. Bi-goal evolution for many-objective optimization problems. Artificial
Intelligence, 2015, 228: 45-65.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'BiGE': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def _estimation(F, r):
    """Bi-goal transformation: proximity (sum of normalised objectives) and crowding degree."""
    N = F.shape[0]
    with np.errstate(all="ignore"):
        F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    fpr = F.sum(axis=1)
    d = np.sqrt(np.maximum(np.sum(F ** 2, 1)[:, None] + np.sum(F ** 2, 1)[None, :] - 2 * F @ F.T, 0.0))
    np.fill_diagonal(d, np.inf)
    case1 = (d < r) & (fpr[:, None] <= fpr[None, :])
    case2 = (d < r) & (fpr[:, None] > fpr[None, :])
    sh = np.zeros((N, N))
    sh[case1] = (0.5 * (1 - d[case1] / r)) ** 2
    sh[case2] = (1.5 * (1 - d[case2] / r)) ** 2
    return np.column_stack([fpr, np.sqrt(sh.sum(axis=1))])


def _environmental_selection(pop, N, rng):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    bi = _estimation(F[last], 1.0 / N ** (1.0 / F.shape[1]))
    front2, max2 = nd_sort(bi, None, N - int(nxt.sum()))
    nxt[last[front2 < max2]] = True
    last2 = last[front2 == max2]
    nxt[last2[rng.permutation(len(last2))[: N - int(nxt.sum())]]] = True
    return pop[nxt]


class BiGE(LoopAlgorithm):
    def step(self):
        N = len(self.pop)
        bi = _estimation(objs(self.pop), 1.0 / self.N ** (1.0 / self.M))
        p1 = self.rng.integers(0, N, size=N)
        p2 = self.rng.integers(0, N, size=N)
        dominate = np.any(bi[p1] < bi[p2], axis=1).astype(int) - np.any(bi[p1] > bi[p2], axis=1).astype(int)
        pool = np.concatenate([p1[dominate >= 0], p2[dominate < 0]])
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N, self.rng)
