# emopylab 2026
"""CMOBR (constrained multiobjective optimization via both constraint and objective relaxations).

Reference:
F. Ming, B. Xue, M. Zhang, W. Gong, and H. Zhen. Constrained multiobjective optimization via
relaxations on both constraints and objectives. IEEE Transactions on Artificial Intelligence, 2024,
5(12): 6709-6722.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness as _cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, neighbors_of, objs, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'CMOBR': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _overall_cv(c):
    return np.sum(np.abs(np.maximum(c, 0)), axis=1) if np.size(c) else np.zeros(len(c))


def _archive(pop, N):
    F = objs(pop)
    fit = _cal_fitness(F, cons(pop))
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(F[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt]


def _copy(pop):
    return pop[np.arange(len(pop))]


class CMOBR(LoopAlgorithm):
    """Push and pull search whose pull stage additionally evolves an unconstrained population (``Population1``)
    that is first left free, then restricted to an adaptive convergence level ``eta``; both feed a shared archive."""

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
        self.cp, self.alpha, self.tao, self.beta = 2, 0.95, 0.05, 0.95
        self.ideal, self.nadir = {}, {}
        self.arch = _archive(self.pop, N)
        self.initialized, self.eta, self.gen = False, 0.0, 1

    def _max_change(self, gen):
        d = 1e-6
        z0 = np.zeros(self.M)
        a, b = self.ideal.get(gen, z0), self.ideal.get(gen - self.last_gen + 1, z0)
        c, e = self.nadir.get(gen, z0), self.nadir.get(gen - self.last_gen + 1, z0)
        rz = np.abs((a - b) / np.maximum(b, d))
        nrz = np.abs((c - e) / np.maximum(e, d))
        return float(np.max(np.concatenate([rz, nrz])))

    def _neighbours(self, i):
        rng, N = self.rng, self.N
        return self.B[i][rng.permutation(self.B.shape[1])] if rng.random() < self.delta else rng.permutation(N)

    def _substep(self, pop, Z, i, cond_fn):
        """One MOEA/D-style replacement of individual ``i``'s mating neighbourhood in ``pop`` (in place)."""
        P = self._neighbours(i)
        X = decs(pop)
        off = self.evaluate(de(self.problem, X[[i]], X[[P[0]]], X[[P[1]]], rng=self.rng))
        fo = objs(off)[0]
        Z = np.minimum(Z, fo)
        Fp = objs(pop[P])
        g_old = np.max(np.abs(Fp - Z) * self.W[P], axis=1)
        g_new = np.max(np.abs(fo - Z) * self.W[P], axis=1)
        cv_old = _overall_cv(cons(pop[P]))
        cv_new = float(_overall_cv(cons(off))[0])
        cond = cond_fn(P, off, fo, Z, g_old, g_new, cv_old, cv_new)
        for j in np.where(cond)[0][: self.nr]:
            pop[P[j]] = off[0]
        return Z

    def step(self):
        N, M, FE, mx = self.N, self.M, self.FE, self.max_FE
        pop = self.pop
        gen = self.gen
        cv = _overall_cv(cons(pop))
        rf = np.sum(cv <= 1e-6) / N
        self.ideal[gen], self.nadir[gen] = self.Z.copy(), objs(pop).max(axis=0)
        if gen >= self.last_gen:
            self.max_change = self._max_change(gen)
        if FE <= 0.9 * mx:
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
        eps = self.eps_k
        if self.stage == 1:
            for i in range(N):
                self.Z = self._substep(pop, self.Z, i, lambda P, off, fo, Z, go, gn, co, cn: go >= gn)
            self.arch = _archive(Population.merge(self.arch, pop), N)
        else:
            if not self.initialized:
                self.pop1 = _copy(pop)
                self.initialized = True
                self.Z1 = objs(self.pop1).min(axis=0)
                self.Znad = objs(self.pop1).max(axis=0)
            pull = lambda P, off, fo, Z, go, gn, co, cn: ((go >= gn) & (((co <= eps) & (cn <= eps)) | (co == cn))) | (cn < co)
            for i in range(N):
                self.Z = self._substep(pop, self.Z, i, pull)
            if FE <= 0.5 * mx:
                for i in range(N):
                    self.Z1 = self._substep(self.pop1, self.Z1, i, lambda P, off, fo, Z, go, gn, co, cn: go >= gn)
            else:
                F = objs(pop)
                with np.errstate(invalid="ignore", divide="ignore"):
                    Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
                    cx = Fn.sum(axis=1)
                    upper, lower = 2 * np.nanmax(cx), 0.5 * np.nanmin(cx)
                if gen <= 0.9 * mx:
                    self.eta = self.beta * self.eta if self.eta >= lower else upper
                else:
                    self.eta = 0.0

                def restricted(P, off, fo, Z, go, gn, co, cn):
                    self.Znad = objs(self.pop1).max(axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        c_new = np.sum((fo - Z) / (self.Znad - Z), axis=0) * np.ones(len(P))
                    return ((((c_new <= self.eta) | (go >= gn)) & (((co <= eps) & (cn <= eps)) | (co == cn))) | (cn < co))

                for i in range(N):
                    self.Z1 = self._substep(self.pop1, self.Z1, i, restricted)
            self.arch = _archive(Population.merge(self.arch, pop, self.pop1), N)
        self.gen += 1
        if self.FE >= mx:
            pop = self.arch
        self.pop = pop
