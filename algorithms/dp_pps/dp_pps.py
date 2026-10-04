# emopylab 2026
"""DP-PPS (tri-population based push and pull search).

Reference:
F. Ming, W. Gong, L. Wang, and C. Lu. A tri-population based co-evolutionary framework for
constrained multi-objective optimization problems. Swarm and Evolutionary Computation, 2022, 70:
101055.
"""

from __future__ import annotations

import numpy as np

from algorithms.c_dpea.c_dpea import env_selection, env_selection_no_con
from algorithms.ccmo.ccmo import _truncation, cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, ga_half, neighbors_of, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'DPPPS': {'constrained', 'multi', 'real'}}


def _ocv(c):
    return np.sum(np.abs(np.maximum(c, 0)), axis=1) if np.size(c) else np.zeros(len(c))


def _archive(pop, N):
    C = cons(pop)
    pop = pop[np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)]
    if len(pop) == 0:
        return pop
    if len(pop) > N:
        fit = cal_fitness(objs(pop), cons(pop))
        nxt = fit < 1
        K = int(nxt.sum()) - N
        if K > 0:
            idx = np.where(nxt)[0]
            nxt[idx[_truncation(objs(pop)[nxt], K)]] = False
        pop = pop[nxt]
    return pop


class DPPPS(LoopAlgorithm):
    """Push-and-pull search (decomposition, epsilon pull stage) combined with a dual-population helper pair that
    supplies extra GA offspring and whose feasible solutions are merged into the final archive."""

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
        self.P1 = self.evaluate(self.random_decs(N))
        self.P2 = self.evaluate(self.random_decs(N))
        self.alpha_c = 2.0 / (1 + np.exp(-self.FE * 10 / (3 * self.max_FE))) - 1
        self.para = np.ceil(self.max_FE / N) / 2 - np.ceil(self.FE / (3 * N))
        self.Tc = 0.9 * np.ceil(self.max_FE / N)
        self.last_gen, self.thr = 20, 1e-1
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
        return float(np.max(np.concatenate([np.abs((a - b) / np.maximum(b, d)), np.abs((c - e) / np.maximum(e, d))])))

    def step(self):
        N, rng, W, B, pr = self.N, self.rng, self.W, self.B, self.problem
        pop = self.pop
        gen = int(np.ceil(self.FE / (2 * N)))
        cv = _ocv(cons(pop))
        rf = np.sum(cv <= 1e-6) / N
        self.ideal[gen], self.nadir[gen] = self.Z.copy(), objs(pop).max(axis=0)
        if gen >= self.last_gen:
            self.max_change = self._max_change(gen)
        if gen < self.Tc:
            if self.max_change <= self.thr and self.stage == 1:
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
            off = self.evaluate(de(pr, X[[i]], X[[P[0]]], X[[P[1]]], rng=rng))
            fo = objs(off)[0]
            self.Z = np.minimum(self.Z, fo)
            g_old = np.max(np.abs(objs(pop[P]) - self.Z) * W[P], axis=1)
            g_new = np.max(np.abs(fo - self.Z) * W[P], axis=1)
            cv_old = _ocv(cons(pop[P]))
            cv_new = float(_ocv(cons(off))[0])
            if self.stage == 1:
                cond = g_old >= g_new
            else:
                e = self.eps_k
                cond = ((g_old >= g_new) & (((cv_old <= e) & (cv_new <= e)) | (cv_old == cv_new))) | (cv_new < cv_old)
            for j in np.where(cond)[0][: self.nr]:
                pop[P[j]] = off[0]
        self.P1 = self.P1[rng.permutation(N)]
        self.P2 = self.P2[rng.permutation(N)]
        f1 = {tuple(r) for r in objs(self.P1)}
        gamma = 1 - np.sum([tuple(r) in f1 for r in objs(self.P2)]) / N
        self.P1, r1 = env_selection(self.P1, N, self.alpha_c)
        self.P2, r2 = env_selection_no_con(self.P2, N, self.alpha_c, gamma, self.para)
        allp = Population.merge(self.P1, self.P2)
        pool = tournament(2, 2 * N, np.concatenate([r1, r2]), rng=rng)
        off = self.evaluate(ga_half(pr, decs(allp[pool]), rng=rng))
        self.alpha_c = 2.0 / (1 + np.exp(-self.FE * 5 / self.max_FE)) - 1
        self.para = np.ceil(self.max_FE / N) / 2 - np.ceil(self.FE / (2 * N))
        self.P1, _ = env_selection(Population.merge(self.P1, off), N, self.alpha_c)
        self.P2, _ = env_selection_no_con(Population.merge(self.P2, off), N, self.alpha_c, gamma, self.para)
        self.arch = _archive(Population.merge(self.arch, pop), N)
        if self.FE >= self.max_FE:
            fin = _archive(Population.merge(self.arch, self.P1), N)
            self.arch = fin
            if len(fin):
                pop = fin
        self.pop = pop
