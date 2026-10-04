# emopylab 2026
"""DBEMTO (double-balanced evolutionary multi-task optimization).

Reference:
K. Qiao, J. Liang, K. Yu, M. Wang, B. Qu, C. Yue, and Y. Gou. A self-adaptive evolutionary multi-
task based constrained multi-objective evolutionary algorithm. IEEE Transactions on Emerging Topics
in Computational Intelligence, 2023, 7(4): 1098-1112.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, objs, tournament
from algorithms.community_utils.spea import cal_fitness, truncation
from core.population import Population

ALGORITHM_FLAGS = {'DBEMTO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}

FM = np.array([0.6, 0.8, 1.0])
CRM = np.array([0.1, 0.2, 1.0])


def _select(pop, N, constrained):
    F = objs(pop)
    fit = cal_fitness(F, cons(pop)) if constrained else cal_fitness(F)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    succ = nxt[N: 2 * N].astype(float)
    pop, fit = pop[nxt], fit[nxt]
    order = np.argsort(fit, kind="stable")
    return pop[order], succ, fit[order]


def _gn_r1r2r3(rng, NP1, r0):
    """Random index vectors (1-based like the reference) with r1 != r0, r2 not in {r0, r1}, r3 not in {r0, r1, r2}."""
    n0 = len(r0)
    r1 = np.floor(rng.random(n0) * NP1).astype(int) + 1
    for _ in range(1001):
        pos = r1 == r0
        if not pos.any():
            break
        r1[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int) + 1
    r2 = np.floor(rng.random(n0) * NP1).astype(int) + 1
    for _ in range(1001):
        pos = (r2 == r1) | (r2 == r0)
        if not pos.any():
            break
        r2[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int) + 1
    r3 = np.floor(rng.random(n0) * NP1).astype(int) + 1
    for _ in range(1001):
        pos = (r3 == r1) | (r3 == r0) | (r3 == r2)
        if not pos.any():
            break
        r3[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int) + 1
    return r1, r2, r3


class DBEMTO(LoopAlgorithm):
    """Two populations (task 1 with the constraints, task 2 without) breed each generation with three operators — a genetic
    one, differential evolution and a transfer of the other task's solutions; the share of individuals given to each is set
    by which operator produced the most surviving offspring in the previous generation (at least 5% each)."""

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(self.pop_size))

    def _initialize_advance(self, infills=None, **kwargs):
        self.tasks = [infills, self.evaluate(self.random_decs(self.pop_size))]
        self.fit = [cal_fitness(objs(self.tasks[0]), cons(self.tasks[0])), cal_fitness(objs(self.tasks[1]))]
        self.change = np.ones((2, 3))
        self.best = [2, 2]
        self.pop = self.tasks[0]
        self._set_optimum()

    def _breed(self, i, val_off):
        rng, N, D = self.rng, self.N, self.D
        rate = self.change[i].copy()
        score = np.where(np.isnan(rate), -np.inf, rate)
        self.best[i] = int(np.argmax(score)) + 1
        if np.sum(rate == rate[0]) == 3:
            self.best[i] = 1
        self.change[i] = 0.1
        perm = rng.permutation(N) + 1                                   # values 1..N as in the reference
        a, b = int(np.ceil(0.05 * N)), int(np.ceil(2 * 0.05 * N))
        blk1, blk2, rest = perm[:a], perm[int(np.ceil(0.05 * N + 1)) - 1: b], perm[int(np.ceil(2 * 0.05 * N + 1)) - 1:]
        if self.best[i] == 1:
            third, second, first = blk1, blk2, rest
        elif self.best[i] == 2:
            third, first, second = blk1, blk2, rest
        else:
            first, second, third = blk1, blk2, rest
        pop = self.tasks[i]
        X = decs(pop)
        slots = [None] * N
        if len(first):
            mate = tournament(2, 2 * len(first), self.fit[i], rng=rng)
            new = self.evaluate(ga_half(self.problem, X[mate], rng=rng))
            for pos, ind in zip(first, new):
                slots[pos - 1] = ind
        if len(second):
            r1, r2, r3 = _gn_r1r2r3(rng, N, perm)
            idx = second - 1
            n2 = len(second)
            F2 = FM[rng.integers(0, 3, n2)][:, None]
            CR2 = CRM[rng.integers(0, 3, n2)][:, None]
            pop2 = X[idx]
            vi = X[r1[idx] - 1] + F2 * (X[r2[idx] - 1] - X[r3[idx] - 1])
            lo, up = self.lower, self.upper
            below = vi < lo
            vi = np.where(below, (pop2 + lo) / 2, vi)
            above = vi > up
            vi = np.where(above, (pop2 + up) / 2, vi)
            mask = rng.random((n2, D)) > CR2
            mask[np.arange(n2), np.floor(rng.random(n2) * D).astype(int)] = False
            u = np.where(mask, pop2, vi)
            new = self.evaluate(u)
            for pos, ind in zip(second, new):
                slots[pos - 1] = ind
        if len(third):
            other = self.tasks[1 - i]
            pick = rng.permutation(N)[: len(third)]
            for pos, k in zip(third, pick):
                slots[pos - 1] = other[k]
        val_off[i] = Population(slots)
        return first, second, third

    def step(self):
        N = self.N
        val_off = [None, None]
        sets = [self._breed(0, val_off), self._breed(1, val_off)]
        for i in range(2):
            constrained = i == 0
            if self.best[i] != 3:
                self.tasks[i], _, self.fit[i] = _select(Population.merge(self.tasks[i], val_off[1 - i]), N, constrained)
            self.tasks[i], succ, self.fit[i] = _select(Population.merge(self.tasks[i], val_off[i]), N, constrained)
            for k, s in enumerate(sets[i]):
                with np.errstate(all="ignore"):
                    self.change[i, k] += np.float64(succ[s - 1].sum()) / np.float64(len(s))
        self.pop = self.tasks[0]
