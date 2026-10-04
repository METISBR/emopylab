# emopylab 2026
"""MOCGDE (multi-objective conjugate gradient and differential evolution algorithm).

Reference:
Y. Tian, H. Chen, H. Ma, X. Zhang, K. C. Tan, and Y. Jin. Integrating conjugate gradients into
evolutionary algorithms for large-scale continuous multi-objective optimization. IEEE/CAA Journal of
Automatica Sinica, 2022, 9(10): 1801-1817.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cv, decs, first_front, gradient_direction, objs, pdist2, cons, truncate_lexi, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MOCGDE': {'constrained', 'integer', 'multi', 'real'}}


def _update_archive(P, N):
    c = cons(P)
    P = P[first_front(objs(P), c if c.size else None)]
    if len(P) > N:
        dis = pdist2(objs(P), objs(P))
        np.fill_diagonal(dis, np.inf)
        P = P[~truncate_lexi(dis, len(P) - N)]
    return P


class MOCGDE(LoopAlgorithm):
    """Multi-objective conjugate-gradient / differential mix: gradient steps on the non-conflicting variables and
    archive differences on the conflicting ones (Fletcher-Reeves directions per subproblem)."""

    def __init__(self, pop_size: int = 100, NP: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.NP = int(NP)

    def initial_size(self):
        self.n_pop_target = self.pop_size
        self.W, sub = uniform_point(self.NP, self.M)
        self.subN = sub
        return sub

    def start(self):
        self.swarm = self.pop
        self.archive = self.pop
        self.K = np.zeros(self.subN, dtype=int)
        self.g0 = [None] * self.subN
        self.d0 = [None] * self.subN
        self.pop = self.archive

    def step(self):
        rng, D = self.rng, self.D
        self.K = self.K % D + 1
        off_pop = []
        for i in range(self.subN):
            gk, site = gradient_direction(self, self.swarm[i], self.W[i])
            if self.K[i] == 1:
                dk = -gk
            else:
                beta = (gk @ gk) / (self.g0[i] @ self.g0[i])
                dk = -gk + beta * self.d0[i]
                if gk @ dk >= 0:
                    dk = -gk
            success = False
            cvp = float(cv(self.swarm[i:i + 1])[0])
            fp = objs(self.swarm[i:i + 1])[0]
            x0 = np.asarray(self.swarm[i].X, dtype=float)
            for step in range(10):
                with np.errstate(all="ignore"):
                    mu = rng.random(D) < 1.0 / max(site.sum(), 1e-300)
                a, b = self.archive[int(rng.integers(0, len(self.archive)))], self.archive[int(rng.integers(0, len(self.archive)))]
                dec = x0 + (~site) * 0.5 ** step * dk + mu * site * 0.5 ** step * (np.asarray(a.X, float) - np.asarray(b.X, float))
                child = self.evaluate(dec[None, :])
                off_pop.append(child[0])
                cvo = float(cv(child)[0])
                if cvo < cvp or (cvo == cvp and np.all(objs(child)[0] < fp)):
                    success = True
                    break
            if success:
                self.swarm[i] = child[0]
                self.g0[i], self.d0[i] = gk, dk
            else:
                self.swarm[i] = self.archive[int(rng.integers(0, len(self.archive)))]
                self.K[i] = 0
        self.archive = _update_archive(Population.merge(self.archive, Population.create(off_pop)), self.n_pop_target)
        self.pop = self.archive
