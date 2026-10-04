# emopylab 2026
"""MOSD (multiobjective steepest descent).

Reference:
X. Liu and A. C. Reynolds. A multiobjective steepest descent method with applications to optimal
well control. Computational Geosciences, 2016, 20: 355-374.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, first_front, objs, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'MOSD': {'constrained', 'large', 'multi', 'real'}}


def _update_archive(P, N):
    P = P[first_front(objs(P), cons(P) if cons(P).size else None)]
    if len(P) > N:
        choose = np.ones(len(P), bool)
        dis = pdist2(objs(P), objs(P))
        np.fill_diagonal(dis, np.inf)
        while choose.sum() > N:
            remain = np.where(choose)[0]
            temp = np.sort(dis[np.ix_(remain, remain)], axis=1)
            choose[remain[np.lexsort(temp.T[::-1])[0]]] = False
        P = P[choose]
    return P


class MOSD(LoopAlgorithm):
    """Bi-objective steepest descent; the objective gradients are forward finite differences
    (``D + 1`` evaluations each), as in the reference implementation."""

    def __init__(self, pop_size: int = 100, step: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.step0 = float(step)

    def start(self):
        self.archive = _update_archive(self.pop, self.N)
        self.step_size = self.step0
        self.pop, self.swarm = self.archive, self.pop

    def step(self):
        pop = self.swarm
        for i in range(self.N):
            og, cg = self.cal_grad(decs(pop[i:i + 1])[0])
            infeasible = bool(np.any(cons(pop[i:i + 1]) > 0))
            gk = (cg if infeasible and cg.shape[0] >= 2 else og).T            # (D, 2)
            g1, g2 = gk[:, 0], gk[:, 1]
            gs, gl = (g1, g2) if np.linalg.norm(g1) <= np.linalg.norm(g2) else (g2, g1)
            if gs @ (gs - gl) <= 0:
                d = -gs / np.linalg.norm(gs)
            else:
                Dv = (((gl @ gl - gs @ gl) / (gs @ gs - gs @ gl)) * (-gs)) - gl
                d = Dv / np.linalg.norm(Dv)
            off = self.evaluate((decs(pop[i:i + 1])[0] + self.step_size * d)[None, :])
            self.archive = _update_archive(Population.merge(self.archive, off), self.N)
            if not np.any(objs(off)[0] < objs(pop[i:i + 1])[0]):
                pop[i] = self.archive[int(self.rng.integers(0, len(self.archive)))]
                self.step_size /= 2
            else:
                pop[i] = off[0]
                self.step_size = min(2 * self.step_size, self.step0)
        self.pop = self.archive
