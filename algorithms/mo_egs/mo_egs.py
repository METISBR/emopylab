# emopylab 2026
"""MO-EGS (multi-objective evolutionary gradient search).

Reference:
C. K. Goh, Y. S. Ong, K. C. Tan, and E. J. Teoh. An investigation on evolutionary gradient search
for multi-objective optimization. Proceedings of the IEEE Congress on Evolutionary Computation,
2008.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, first_front, objs, pdist2, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'MOEGS': {'large', 'multi', 'real'}}


def _cvs(pop):
    C = cons(pop)
    return np.sum(np.maximum(C, 0), axis=1) if C.size else np.zeros(len(pop))


def update_archive(P, N):
    C = cons(P)
    P = P[first_front(objs(P), C if C.size else None)]
    if len(P) > N:
        d = pdist2(objs(P), objs(P))
        np.fill_diagonal(d, np.inf)
        P = P[~truncate_lexi(d, len(P) - N)]
    return P


class MOEGS(LoopAlgorithm):
    """Multi-objective evolutionary gradient search: every parent samples ``r`` Gaussian perturbations, the
    objective (or violation) differences are turned into a stochastic gradient estimate, and the parent follows it
    with a step size ``thao`` that grows with success and shrinks with failure."""

    def __init__(self, pop_size: int = 100, r: int = 500, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.r = int(r)

    def start(self):
        self.own = self.pop
        self.archive = update_archive(self.pop, self.N)
        self.thao = 0.1
        self.pop = self.archive

    def step(self):
        N, r, D, M, rng = self.N, self.r, self.D, self.M, self.rng
        pop = self.own
        for i in range(N):
            parent = pop[[i]]
            pdec, pobj, pcon = np.asarray(parent[0].X, float), objs(parent)[0], None
            pc = cons(parent)
            pcon = pc[0] if pc.size else np.zeros(0)
            z = rng.normal(0.0, self.thao, size=(r, D))
            off = self.evaluate(pdec + z)
            self.archive = update_archive(Population.merge(self.archive, off), N)
            oc = cons(off)
            if np.any(pcon > 0):
                gk = np.sum(oc - pcon, axis=1)
            else:
                w = rng.random((r, 1))
                if M == 2:
                    W = np.hstack([w, 1 - w])
                elif M == 3:
                    W = np.hstack([w / 3, (1 - w / 3) / 2, 1 - w / 3 - (1 - w / 3) / 2])
                else:                                          # generalisation for more than three objectives
                    W = rng.dirichlet(np.ones(M), size=r)
                gk = np.sum(W * (objs(off) - pobj), axis=1)
            g = gk @ z
            with np.errstate(all="ignore"):
                c = pdec - self.thao * g / np.linalg.norm(g)
            child = self.evaluate(c[None, :])
            cvo = float(_cvs(child)[0])
            cvp = float(np.sum(np.maximum(pcon, 0)))
            if cvo < cvp or np.all(objs(child)[0] < pobj):
                pop[i] = child[0]
                self.thao *= 1.8
            elif cvp < cvo or np.all(pobj < objs(child)[0]):
                pop[i] = self.archive[int(rng.integers(len(self.archive)))]
                self.thao /= 1.8
            elif rng.random() > 0.5:
                pop[i] = child[0]
            else:
                pop[i] = self.archive[int(rng.integers(len(self.archive)))]
            self.archive = update_archive(Population.merge(self.archive, child), N)
            if self.thao > 1.2:
                self.thao /= 1.2
            elif self.thao < 1e-4:
                self.thao *= 1.2
        self.own = pop
        self.pop = self.archive
