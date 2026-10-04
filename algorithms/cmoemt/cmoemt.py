# emopylab 2026
"""CMOEMT (constrained multi-objective optimization based on evolutionary multitasking optimization).

Reference:
F. Ming, W. Gong, L. Wang, and L. Gao. Constrained multi-objective optimization via multitasking and
knowledge transfer. IEEE Transactions on Evolutionary Computation, 2024, 28(1): 77-89.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import cal_fitness, environmental_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, ga_half, neighbors_of, objs, uniform_point
from algorithms.community_utils.base import tournament
from core.population import Population

ALGORITHM_FLAGS = {'CMOEMT': {'constrained', 'multi', 'real'}}


def _ocv(c):
    return np.sum(np.abs(np.maximum(c, 0)), axis=1) if np.size(c) else np.zeros(len(c))


class CMOEMT(LoopAlgorithm):
    """Constrained evolutionary multitasking with three tasks: the constrained main task, a decomposition-based
    push-and-pull helper (unconstrained first, epsilon-constrained afterwards) and an unconstrained helper; the
    first half of the run selects the main and third populations from the offspring only, the second half also
    lets the three parent populations compete."""

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
        self.P = [self.pop, self.evaluate(self.random_decs(N))]
        self.Z = objs(self.P[1]).min(axis=0)
        self.P.append(self.evaluate(self.random_decs(N)))
        self.f1 = cal_fitness(objs(self.P[0]), cons(self.P[0]))
        self.f3 = cal_fitness(objs(self.P[2]))
        G = int(np.ceil(self.max_FE / N))
        self.Tc, self.last_gen, self.thr = 0.9 * G, 20, 1e-1
        self.stage, self.max_change = 1, 1.0
        self.eps_k = self.eps_0 = 0.0
        self.cp, self.alpha, self.tao = 2, 0.95, 0.05
        self.ideal, self.nadir = {}, {}

    def _max_change(self, gen):
        d = 1e-6
        z0 = np.zeros(self.M)
        a, b = self.ideal.get(gen, z0), self.ideal.get(gen - self.last_gen + 1, z0)
        c, e = self.nadir.get(gen, z0), self.nadir.get(gen - self.last_gen + 1, z0)
        return float(np.max(np.concatenate([np.abs((a - b) / np.maximum(b, d)), np.abs((c - e) / np.maximum(e, d))])))

    def _helper_offspring(self):
        rng, N, W, B, M = self.rng, self.N, self.W, self.B, self.M
        P2 = self.P[1]
        out = []
        for _ in range(5):
            boundary = np.where(np.sum(W < 1e-3, axis=1) == M - 1)[0]
            boundary = np.concatenate([boundary, [len(W) // 2]])
            I = np.concatenate([boundary, rng.integers(0, len(W), size=max(N // 5 - len(boundary), 0))]).astype(int)
            for i in I:
                P = B[i][rng.permutation(B.shape[1])] if rng.random() < self.delta else rng.permutation(N)
                X = decs(P2)
                off = self.evaluate(de(self.problem, X[[i]], X[[P[0]]], X[[P[1]]], rng=rng))
                out.append(off[0])
                fo = objs(off)[0]
                self.Z = np.minimum(self.Z, fo)
                g_old = np.max(np.abs(objs(P2[P]) - self.Z) * W[P], axis=1)
                g_new = np.max(np.abs(fo - self.Z) * W[P], axis=1)
                cv_old = _ocv(cons(P2[P]))
                cv_new = float(_ocv(cons(off))[0])
                if self.stage == 1:
                    cond = g_old >= g_new
                else:
                    e = self.eps_k
                    cond = ((g_old >= g_new) & (((cv_old <= e) & (cv_new <= e)) | (cv_old == cv_new))) | (cv_new < cv_old)
                for j in np.where(cond)[0][: self.nr]:
                    P2[P[j]] = off[0]
        return Population.create(out)

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        first_half = self.FE < self.max_FE / 2
        gen = int(np.ceil(self.FE / (2 * N)))
        P2 = self.P[1]
        cv = _ocv(cons(P2))
        F2 = objs(P2)
        rf = np.sum(cv <= 1e-6) / N
        self.ideal[gen], self.nadir[gen] = self.Z.copy(), F2.max(axis=0)
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
        off1 = self.evaluate(ga_half(pr, decs(self.P[0][tournament(2, N, self.f1, rng=rng)]), rng=rng))
        off2 = self._helper_offspring()
        off3 = self.evaluate(ga_half(pr, decs(self.P[2][tournament(2, N, self.f3, rng=rng)]), rng=rng))
        if first_half:
            pool = Population.merge(self.P[0], off1, off2, off3)
            pool3 = Population.merge(self.P[2], off1, off2, off3)
        else:
            pool = Population.merge(self.P[0], self.P[1], self.P[2], off1, off2, off3)
            pool3 = pool
        self.P[0], self.f1 = environmental_selection(pool, N, True)
        self.P[2], self.f3 = environmental_selection(pool3, N, False)
        self.pop = self.P[0]

