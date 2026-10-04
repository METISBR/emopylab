# emopylab 2026
"""VaEA (vector angle based evolutionary algorithm).

Reference:
Y. Xiang, Y. Zhou, M. Li, and Z. Chen. A vector angle-based evolutionary algorithm for unconstrained
many-objective optimization. IEEE Transactions on Evolutionary Computation, 2017, 21(1): 131-152.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, angle_matrix, decs, ga, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'VaEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _association(F1, F2, N):
    """Boolean selection mask over [F1; F2]: F1 all kept, F2 filled by maximum-vector-angle rule."""
    N1, N2, M = len(F1), len(F2), F2.shape[1]
    F = np.vstack([F1, F2])
    with np.errstate(all="ignore"):
        F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    F = np.nan_to_num(F)
    fit = F.sum(axis=1)
    angle = angle_matrix(F)
    choose = np.concatenate([np.ones(N1, bool), np.zeros(N2, bool)])
    if not choose.any():
        cosd = 1.0 - np.cos(angle_matrix(F2, np.eye(M)))
        choose[N1 + np.argmin(cosd, axis=0)] = True
        rank = np.argsort(fit[N1:], kind="stable")
        choose[N1 + rank[: min(M, len(rank))]] = True
    while choose.sum() < N:
        select = np.where(choose)[0]
        remain = np.where(~choose)[0]
        rho = np.argmax(angle[np.ix_(remain, select)].min(axis=1))
        choose[remain[rho]] = True
        if not choose.all():
            select = np.append(select, remain[rho])
            remain = np.delete(remain, rho)
            sub = angle[np.ix_(remain, select)]
            mu = np.argmin(sub.min(axis=1))
            r = np.argmin(sub[mu])
            theta = sub[mu, r]
            if theta < np.pi / 2 / (N + 1) and fit[select[r]] > fit[remain[mu]]:
                choose[select[r]] = False
                choose[remain[mu]] = True
    return choose


def _environmental_selection(pop, N):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = np.concatenate([np.where(front_no < max_f)[0], np.where(front_no == max_f)[0]])
    choose = _association(F[front_no < max_f], F[front_no == max_f], N)
    return pop[nxt[choose]]


class VaEA(LoopAlgorithm):
    def step(self):
        pool = self.rng.integers(0, len(self.pop), size=self.N)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N)
