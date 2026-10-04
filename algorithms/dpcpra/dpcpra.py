# emopylab 2026
"""DPCPRA (dual-population with dynamic constraint processing and resource allocating).

Reference:
K. Qiao, Z. Chen, B. Qu, K. Yu, C. Yue, K. Chen, and J. Liang. A dual- population evolutionary
algorithm based on dynamic constraint processing and resources allocation for constrained multi-
objective optimization problems. Expert Systems With Applications, 2024, 238: 121707.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness as cal_fitness_pop1
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, objs, tournament
from algorithms.mscmo.mscmo import _con, archive_fill, archive_strict, cal_fitness as cal_fitness_pop2
from core.population import Population

ALGORITHM_FLAGS = {'DPCPRA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _select(pop, fit, N):
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return nxt


def _finish(pop, fit, nxt, N, n_own):
    rate = float(np.sum(nxt[N:N + n_own]) / n_own) if n_own else 0.0
    rate = min(max(rate, 0.1), 0.9)
    p, f = pop[nxt], fit[nxt]
    r = np.argsort(f, kind="stable")
    return p[r], f[r], rate


class DPCPRA(LoopAlgorithm):
    """Dual-population constrained algorithm with priority-driven constraint relaxation: the second population
    adds the constraints one by one (hardest first) once its objectives settle; the population sizes used for
    breeding follow the success rate of each population's offspring."""

    def start(self):
        N = self.N
        self.P1 = self.pop
        self.P2 = self.evaluate(self.random_decs(N))
        self.f1 = cal_fitness_pop1(objs(self.P1), cons(self.P1))
        self.current, self.gen, self.last_gen, self.thr = 0, 0, 100, 1e-2
        self.change = {}
        self.priority = np.array([], dtype=int)
        self.flag, self.handling = 0, 0
        self.archive = self.P2
        self.f2 = cal_fitness_pop2(objs(self.P2), _con(self.P2), self.priority, 0, 0)
        self.sr1 = 0.5
        self.feasible_rate = None

    def _normalize(self, G):
        F = objs(self.P2)
        with np.errstate(all="ignore"):
            self.change[G] = np.mean((F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0)), axis=0)

    def _converged(self, G):
        if G - self.gen > self.last_gen:
            prev = self.change.get(G - self.last_gen, np.zeros(self.M))
            return bool(np.max(np.abs(self.change[G] - prev)) <= self.thr)
        return False

    def _stage_control(self, G):
        N = self.N
        nCon = _con(self.P2).shape[1]
        if self.flag == 0:
            self._normalize(G)
            if self._converged(G):
                self.flag = 1
                C = _con(self.P2)
                self.feasible_rate = np.array([np.sum(C[:, j] <= 0) / len(self.P2) for j in range(nCon)])
                self.priority = np.argsort(self.feasible_rate, kind="stable")
                self.P2 = self.evaluate(self.random_decs(N))
            return
        if self.current == 0:
            if np.mean(_con(self.P2)[:, self.priority[0]] > 0) > 0:
                self.current += 1
                self.gen = G + 1
        elif self.current <= nCon and self.handling == 0:
            self._normalize(G)
            if self._converged(G):
                if self.current < nCon and self.feasible_rate[self.priority[self.current]] != 1:
                    self.current += 1
                elif self.current < nCon and self.feasible_rate[self.priority[self.current]] == 1:
                    self.current = nCon
                elif self.current == nCon:
                    self.handling = 1
                if len(self.archive) != N:
                    self.archive = archive_fill(Population.merge(self.archive, self.P2), N, len(self.archive))
                self.P2 = self.archive
                self.f2 = cal_fitness_pop2(objs(self.P2), _con(self.P2), self.priority, self.current, self.handling)
                self.gen = G + 1

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        G = int(np.ceil(self.FE / N))
        self._stage_control(G)
        P1, P2 = self.P1, self.P2
        if self.flag == 0:
            off1 = self.evaluate(ga_half(pr, decs(P1[tournament(2, N, self.f1, rng=rng)]), rng=rng))
            off2 = self.evaluate(ga_half(pr, decs(P2[tournament(2, N, self.f2, rng=rng)]), rng=rng))
        else:
            k = int(np.ceil(N * self.sr1))
            k = k + 1 if k % 2 else k
            off1 = self.evaluate(ga_half(pr, decs(P1[tournament(2, k, self.f1, rng=rng)]), rng=rng))
            n2 = N - 2 * len(off1)
            off2 = self.evaluate(ga_half(pr, decs(P2[tournament(2, max(n2, 0), self.f2, rng=rng)]), rng=rng))
        if self.flag == 1 and self.handling != 1:
            self.archive = archive_strict(Population.merge(off2, self.archive), N, self.priority, self.current)
        pool1 = Population.merge(P1, off1, off2)
        fit1 = cal_fitness_pop1(objs(pool1), cons(pool1))
        self.P1, self.f1, s1 = _finish(pool1, fit1, _select(pool1, fit1, N), N, len(off1))
        pool2 = Population.merge(P2, off2, off1)
        fit2 = cal_fitness_pop2(objs(pool2), _con(pool2), self.priority, self.current, self.handling)
        self.P2, self.f2, s2 = _finish(pool2, fit2, _select(pool2, fit2, N), N, len(off2))
        self.sr1 = s1 / (s1 + s2)
        self.pop = self.P1
