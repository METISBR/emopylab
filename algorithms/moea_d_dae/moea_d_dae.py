# emopylab 2026
"""MOEA-D-DAE (mOEA/D with detect-and-escape strategy).

Reference:
Q. Zhu, Q. Zhang, and Q. Lin. A constrained multi-objective evolutionary algorithm with detect-and-
escape strategy. IEEE Transactions on Evolutionary Computation, 2020, 24(5): 938-947.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, cons, crowding, decs, first_front, ga_half, nd_sort, objs, pdist2,
                                            tournament, uniform_point)
from core.population import Population

ALGORITHM_FLAGS = {'MOEADDAE': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _cv(pop):
    C = cons(pop)
    return np.sum(np.maximum(C, 0), axis=1) if C.size else np.zeros(len(pop))


def archive_update(pop, N, rng):
    feas = _cv(pop) <= 0
    pop = pop[feas]
    if len(pop) == 0:
        return pop
    F = objs(pop)
    if F.shape[1] == 2:
        pop = pop[first_front(F)]
        if len(pop) > N:
            cd = crowding(objs(pop))
            pop = pop[np.argsort(-cd, kind="stable")[:N]]
        return pop
    pop = pop[first_front(F)]
    pop = pop[rng.permutation(len(pop))]
    if len(pop) > N:
        P = objs(pop)
        with np.errstate(all="ignore"):
            P = (P - P.min(axis=0)) / (P.max(axis=0) - P.min(axis=0))
        d = pdist2(P, P)
        np.fill_diagonal(d, np.inf)
        sd = np.sort(d, axis=1)
        r = sd[:, min(3, sd.shape[1]) - 1].mean()
        R = np.minimum(d / r, 1.0)
        keep = np.arange(len(pop))
        while len(keep) > N:
            worst = int(np.argmax(1 - np.prod(R, axis=1)))
            keep = np.delete(keep, worst)
            R = np.delete(np.delete(R, worst, axis=0), worst, axis=1)
        pop = pop[keep]
    return pop


class MOEADDAE(LoopAlgorithm):
    """MOEA/D with dynamic adaptive epsilon constraint handling in three stages (explore, relax, return) that tracks
    the rate of change of the total violation, plus an external archive of feasible non-dominated solutions."""

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 5))
        self.B = np.argsort(pdist2(self.W, self.W), axis=1, kind="stable")[:, : self.T]
        return self.pop_size

    def _sigma_min(self, eps):
        return 1 - (1 / (eps + 1.0e-10)) ** (3 / np.ceil(self.max_FE / self.N))

    def start(self):
        N, W = self.N, self.W
        pop = self.pop
        self.P = pop
        self.z = objs(pop).min(axis=0)
        self.Pi = np.ones(N)
        cv = _cv(pop)
        self.CP_old = float(cv.sum())
        self.avg_fit = float(np.sum(np.max(np.abs((objs(pop) - self.z) * W), axis=1)) / N)
        self.eps_max = float(cv.max())
        C = cons(pop)
        self.eps = self.eps_max * (C.shape[1] if C.size else 0)
        self.sigma_min = self._sigma_min(self.eps)
        self.current, self.gen = 0, 0
        self.A = archive_update(pop, N, self.rng)
        self.tA = None
        fr = np.sum(cv <= 0) / N
        self.sigma = max(self.sigma_min, fr)

    def _update(self, P, off, tgt, use_eps):
        """Shared neighbour-replacement loop (``UpdatePop`` when ``tgt`` is the population, ``UpdatetA`` for the
        temporary archive)."""
        rng, W, z = self.rng, self.W, self.z
        fo, cvo = objs(off)[0], float(_cv(off)[0])
        P = list(P)
        c = 0
        while c != 2 and P:
            k = int(rng.permutation(len(P))[0])
            i = P[k]
            g_o = np.max(np.abs(fo - z) * W[i])
            fi, cvi = objs(tgt[[i]])[0], float(_cv(tgt[[i]])[0])
            g_pi = np.max(np.abs(fi - z) * W[i])
            if use_eps:
                if cvi <= self.eps and cvo < self.eps:
                    f_o, f_pi = g_o, g_pi
                else:
                    f_o = self.sigma * g_o + (1 - self.sigma) * self.avg_fit * cvo
                    f_pi = self.sigma * g_pi + (1 - self.sigma) * self.avg_fit * cvi
                better = f_o < f_pi
            else:
                e = 1.0 / len(tgt)
                better = (e * g_o + (1 - e) * self.avg_fit * cvo) < (e * g_pi + (1 - e) * self.avg_fit * cvi)
            if better:
                tgt[i] = off[0]
                if use_eps:
                    delta = (f_pi - f_o) / f_pi
                    self.Pi[i] = 1 if delta > 0.001 else 0.95 + (0.05 * delta / 0.001) * self.Pi[i]
                    c += 1
            if not use_eps:
                c += 1
            del P[k]

    def step(self):
        N, M, rng, W, B = self.N, self.M, self.rng, self.W, self.B
        Q = []
        for _ in range(5):
            boundary = np.where(np.sum(W < 1e-3, axis=1) == M - 1)[0]
            I = np.concatenate([boundary, tournament(10, N // 5 - len(boundary), -self.Pi, rng=rng)]).astype(int)
            for i in I:
                P = B[i][rng.permutation(B.shape[1])] if rng.random() < 0.9 else rng.permutation(N)
                off = self.evaluate(ga_half(self.problem, decs(self.P[P[:2]]), rng=rng))
                self.z = np.minimum(self.z, objs(off)[0])
                self._update(P, off, self.P, True)
                if self.current == 1:
                    self._update(P, off, self.tA, False)
                if _cv(off)[0] == 0:
                    Q.append(off[0])
        self.gen += 1
        cv = _cv(self.P)
        fr = np.sum(cv <= 0) / N
        self.sigma = max(self.sigma_min, fr)
        if self.current in (0, 2):
            if fr < 0.95:
                self.eps = (1 - self.sigma) * self.eps
            elif self.current == 0:
                self.current, self.eps, self.tA, self.gen = 1, 1e30, self.P.copy(deep=False), 1
            else:
                self.eps = self.eps_max
                self.sigma_min = self._sigma_min(self.eps)
        if self.gen % 10 == 0:
            CP = float(np.sum(_cv(self.P)))
            ROC = abs(CP - self.CP_old) / (CP + 1e-10)
            self.CP_old = CP
            self.avg_fit = float(np.sum(np.max(np.abs((objs(self.P) - self.z) * W), axis=1)) / N)
            if self.current == 0:
                if 0 < ROC < 1e-5 and CP > 0.1 * self.eps_max:
                    self.current, self.eps, self.gen, self.tA = 1, 1e30, 1, self.P.copy(deep=False)
            elif self.current == 1:
                if 0 < ROC < 1e-5:
                    self.current = 2
                    self.P = self.tA
                    self.eps_max = float(_cv(self.P).max())
                    self.eps = self.eps_max
                    self.sigma_min = self._sigma_min(self.eps)
                    self.z = objs(self.P).min(axis=0)
        if Q:
            self.A = archive_update(Population.merge(self.A, Population.create(Q)), N, rng)
        self.pop = self.A if len(self.A) else self.P
