# emopylab 2026
"""PPS (push and pull search algorithm).

Reference:
Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, and E. Goodman. Push and pull search for
solving constrained multi-objective optimization problems. Swarm and Evolutionary Computation, 2019,
44(2): 665-679.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, cons, decs, de, first_front, neighbors_of, objs, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'PPS': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _overall_cv(c):
    return np.sum(np.abs(np.maximum(c, 0)), axis=1) if np.size(c) else np.zeros(len(c))


def _archive(pop, N):
    c = cons(pop)
    feas = np.all(c <= 0, axis=1) if c.size else np.ones(len(pop), bool)
    pop = pop[feas]
    if len(pop) == 0:
        return pop
    pop = pop[first_front(objs(pop))]
    if len(pop) > N:
        cd = crowding(objs(pop))
        pop = pop[np.argsort(-cd, kind="stable")[:N]]
    return pop


class PPS(LoopAlgorithm):
    """Push and pull search: the unconstrained push stage runs until the ideal/nadir points stabilise, then an
    adaptively shrinking epsilon-constraint pull stage enforces feasibility."""

    def __init__(self, pop_size: int = 100, delta: float = 0.9, nr: int = 2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta, self.nr = float(delta), int(nr)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        N = self.N
        self.Z = objs(self.pop).min(axis=0)
        self.Tc = 0.9 * np.ceil(self.max_FE / N)
        self.last_gen, self.change_threshold = 20, 1e-1
        self.stage, self.max_change = 1, 1.0
        self.eps_k = self.eps_0 = 0.0
        self.cp, self.alpha, self.tao = 2, 0.95, 0.05
        self.ideal, self.nadir = {}, {}
        self.arch = _archive(self.pop, N)

    def _max_change(self, gen):
        d = 1e-6
        z0 = np.zeros(self.M)
        a, b = self.ideal.get(gen, z0), self.ideal.get(gen - self.last_gen + 1, z0)
        c, e = self.nadir.get(gen, z0), self.nadir.get(gen - self.last_gen + 1, z0)
        rz = np.abs((a - b) / np.maximum(b, d))
        nrz = np.abs((c - e) / np.maximum(e, d))
        return float(np.max(np.concatenate([rz, nrz])))

    def step(self):
        N, rng, W, B = self.N, self.rng, self.W, self.B
        pop = self.pop
        gen = int(np.ceil(self.FE / N))
        cv = _overall_cv(cons(pop))
        F = objs(pop)
        rf = np.sum(cv <= 1e-6) / N
        self.ideal[gen], self.nadir[gen] = self.Z.copy(), F.max(axis=0)
        if gen >= self.last_gen:
            self.max_change = self._max_change(gen)
        if gen < self.Tc:
            if self.max_change <= self.change_threshold and self.stage == 1:
                self.stage = -1
                self.eps_0 = float(cv.max())
                self.eps_k = self.eps_0
            if self.stage == -1:
                if rf < self.alpha:
                    self.eps_k = (1 - self.tao) * self.eps_k
                else:
                    self.eps_k = self.eps_0 * ((1 - gen / self.Tc) ** self.cp)
        else:
            self.eps_k = 0.0
        for i in range(N):
            P = B[i][rng.permutation(B.shape[1])] if rng.random() < self.delta else rng.permutation(N)
            X = decs(pop)
            off = self.evaluate(de(self.problem, X[[i]], X[[P[0]]], X[[P[1]]], rng=rng))
            fo = objs(off)[0]
            self.Z = np.minimum(self.Z, fo)
            Fp = objs(pop[P])
            g_old = np.max(np.abs(Fp - self.Z) * W[P], axis=1)
            g_new = np.max(np.abs(fo - self.Z) * W[P], axis=1)
            cv_old = _overall_cv(cons(pop[P]))
            cv_new = float(_overall_cv(cons(off))[0])
            if self.stage == 1:
                cond = g_old >= g_new
            else:
                cond = ((g_old >= g_new) & (((cv_old <= self.eps_k) & (cv_new <= self.eps_k)) | (cv_old == cv_new))) | (cv_new < cv_old)
            for j in np.where(cond)[0][: self.nr]:
                pop[P[j]] = off[0]
        self.arch = _archive(Population.merge(self.arch, pop), N)
        if self.FE >= self.max_FE and len(self.arch):
            pop = self.arch
        self.pop = pop
