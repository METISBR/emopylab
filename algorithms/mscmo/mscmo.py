# emopylab 2026
"""MSCMO (multi-stage constrained multi-objective evolutionary algorithm).

Reference:
H. Ma, H. Wei, Y. Tian, R. Cheng, and X. Zhang. A multi-stage evolutionary algorithm for multi-
objective optimization with complex constraints. Information Sciences, 2021, 560: 68-91.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation
from algorithms.community_utils.base import LoopAlgorithm, cons, cosine_distance, decs, de, first_front, ga, objs, pdist2, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MSCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _con(pop):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), 1))


def cal_fitness(F, C, priority, current, handling):
    N = len(F)
    if current == 0:
        CV = np.zeros(N)
    else:
        CV = np.sum(np.maximum(0, C[:, priority[:current]]), axis=1)
    if handling == 0:
        Z = np.column_stack([F, CV])
        k = (Z[:, None, :] < Z[None, :, :]).any(axis=2).astype(int) - (Z[:, None, :] > Z[None, :, :]).any(axis=2).astype(int)
        Dom = k == 1
    else:
        k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
        Dom = (CV[:, None] < CV[None, :]) | ((CV[:, None] == CV[None, :]) & (k == 1))
    R = Dom.sum(axis=1) @ Dom
    d = pdist2(F, F)
    np.fill_diagonal(d, np.inf)
    D = 1.0 / (np.sort(d, axis=1)[:, max(int(np.floor(np.sqrt(N))) - 1, 0)] + 2)
    return R + D


def _truncate_pop(pop, K):
    return _truncation(objs(pop), K)


def env_selection(pop, N, priority, current, handling):
    fit = cal_fitness(objs(pop), _con(pop), priority, current, handling)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    pop, fit = pop[nxt], fit[nxt]
    r = np.argsort(fit, kind="stable")
    return pop[r], fit[r]


def archive_strict(pop, N, priority, count):
    """Feasible (w.r.t. the first ``count`` prioritised constraints) non-dominated solutions, truncated to N."""
    if count == 0:
        pop = pop[[0]]                                    # the reference indexes the population with a scalar mask
    else:
        C = _con(pop)[:, priority[:count]]
        pop = pop[np.all(C <= 0, axis=1)]
    if len(pop) == 0:
        return pop
    pop = pop[first_front(objs(pop))]
    if len(pop) > N:
        pop = pop[~_truncate_pop(pop, len(pop) - N)]
    return pop


def archive_fill(pop, N, feasible_number):
    """Complete the archive to N solutions with one solution per empty weight-vector region at a time."""
    if feasible_number == 0:
        return pop
    feas, remain = pop[:feasible_number], pop[feasible_number:]
    W, _ = uniform_point(N, objs(pop).shape[1])
    itr = 1
    while len(feas) < N and len(remain):
        progressed = False
        for i in range(len(W)):
            if len(feas) == N:
                break
            r1 = np.argmax(1 - cosine_distance(objs(feas), W), axis=1)
            r2 = np.argmax(1 - cosine_distance(objs(remain), W), axis=1)
            a, b = np.where(r1 == i)[0], np.where(r2 == i)[0]
            if len(a) < itr and len(b):
                feas = Population.merge(feas, remain[[b[0]]])
                remain = remain[np.delete(np.arange(len(remain)), b[0])]
                progressed = True
            if len(remain) == 0:
                break
        itr += 1
        if itr > 10 * N and not progressed:
            break
    return feas


class MSCMO(LoopAlgorithm):
    """Multi-stage constrained multi-objective algorithm: the population first ignores the constraints, then the
    constraints are switched on one at a time (hardest first) whenever the average normalised objectives settle,
    and finally full constrained domination is used; a feasible archive bridges the stages."""

    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type

    def start(self):
        self.gen, self.last_gen, self.thr = 0, 100, 1e-2
        self.change = {}
        self.priority, self.current, self.flag, self.handling = np.array([], dtype=int), 0, 0, 0
        self.fit = cal_fitness(objs(self.pop), _con(self.pop), self.priority, 0, 0)
        self.archive = self.pop
        self.feasible_rate = None

    def _normalize(self, G):
        F = objs(self.pop)
        with np.errstate(all="ignore"):
            self.change[G] = np.mean((F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0)), axis=0)

    def _converged(self, G):
        if G - self.gen > self.last_gen:
            prev = self.change.get(G - self.last_gen, np.zeros(self.M))
            return bool(np.max(np.abs(self.change[G] - prev)) <= self.thr)
        return False

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        G = int(np.ceil(self.FE / N))
        pop = self.pop
        nCon = _con(pop).shape[1]
        if self.flag == 0:
            self._normalize(G)
            if self._converged(G):
                self.flag = 1
                C = _con(pop)
                self.feasible_rate = np.array([np.sum(C[:, j] <= 0) / len(pop) for j in range(nCon)])
                self.priority = np.argsort(self.feasible_rate, kind="stable")
                pop = self.evaluate(self.random_decs(N))
        else:
            if self.current == 0:
                if np.mean(_con(pop)[:, self.priority[0]] > 0) > 0:
                    self.current += 1
                    self.gen = G + 1
            elif self.current <= nCon:
                if self.handling == 0:
                    self.pop = pop
                    self._normalize(G)
                    if self._converged(G):
                        if self.current < nCon and self.feasible_rate[self.priority[self.current]] != 1:
                            self.current += 1
                        elif self.current < nCon and self.feasible_rate[self.priority[self.current]] == 1:
                            self.current = nCon
                        elif self.current == nCon:
                            self.handling = 1
                        if len(self.archive) == N:
                            pop = self.archive
                        else:
                            merged = Population.merge(self.archive, pop)
                            self.archive = archive_fill(merged, N, len(self.archive))
                            pop = self.archive
                        self.fit = cal_fitness(objs(pop), _con(pop), self.priority, self.current, self.handling)
                        self.gen = G + 1
        if self.type == 1:
            off = self.evaluate(ga(pr, decs(pop[tournament(2, N, self.fit, rng=rng)]), rng=rng))
        else:
            m1, m2 = tournament(2, N, self.fit, rng=rng), tournament(2, N, self.fit, rng=rng)
            off = self.evaluate(de(pr, decs(pop), decs(pop[m1]), decs(pop[m2]), rng=rng))
        if self.flag == 1 and self.handling != 1:
            self.archive = archive_strict(Population.merge(off, self.archive), N, self.priority, self.current)
        self.pop, self.fit = env_selection(Population.merge(pop, off), N, self.priority, self.current, self.handling)
