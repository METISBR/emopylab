# emopylab 2026
"""EMCMO (evolutionary multitasking-based constrained multiobjective optimization).

Reference:
K. Qiao, K. Yu, B. Qu, J. Liang, H. Song, and C. Yue. An evolutionary multitasking optimization
framework for constrained multi-objective optimization problems. IEEE Transactions on Evolutionary
Computation, 2022, 26(2): 263-277.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'EMCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def environmental_selection(pop, N, is_origin):
    F = objs(pop)
    fit = cal_fitness(F, cons(pop)) if is_origin == 1 else cal_fitness(F)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(F[nxt], int(nxt.sum()) - N)]] = False
    mask = nxt.copy()
    pop, fit = pop[nxt], fit[nxt]
    rank = np.argsort(fit, kind="stable")
    return pop[rank], fit[rank], mask


class EMCMO(LoopAlgorithm):
    """Evolutionary multitasking for constrained problems: the constrained task and its unconstrained helper
    exchange offspring, and after 20% of the budget the transfer is steered by the measured success rate."""

    def start(self):
        self.P = [self.pop, self.evaluate(self.random_decs(self.N))]
        self.fit = [cal_fitness(objs(self.P[0]), cons(self.P[0])), cal_fitness(objs(self.P[1]))]
        self.transfer = 0

    def step(self):
        N, rng, P = self.N, self.rng, self.P
        off = [None, None]
        if self.transfer == 0:
            for i in range(2):
                off[i] = self.evaluate(ga_half(self.problem, decs(P[i][rng.integers(0, N, N)]), rng=rng))
            new0 = environmental_selection(Population.merge(P[0], off[0], off[1]), N, 1)
            new1 = environmental_selection(Population.merge(P[1], off[1], off[0]), N, 2)
            (P[0], self.fit[0], _), (P[1], self.fit[1], _) = new0, new1
            if self.FE / self.max_FE >= 0.2:
                self.transfer = 1
        else:
            for i in range(2):
                off[i] = self.evaluate(ga_half(self.problem, decs(P[i][tournament(2, N, self.fit[i], rng=rng)]), rng=rng))
            succ = [0.0, 0.0]
            _, _, nxt = environmental_selection(Population.merge(P[1], off[1]), N, 1)
            succ[0] = nxt[:N].sum() / 100 - nxt[N:].sum() / 50
            _, _, nxt = environmental_selection(Population.merge(P[0], off[0]), N, 2)
            succ[1] = nxt[:N].sum() / 100 - nxt[N:].sum() / 50
            for i in range(2):
                o = 1 - i
                if succ[i] > 0:
                    pick = rng.permutation(N)[: N // 2]
                    merged = Population.merge(P[i], off[i], P[o][pick])
                else:
                    merged = Population.merge(P[i], off[i], off[o])
                P[i], self.fit[i], _ = environmental_selection(merged, N, i + 1)
        self.pop = P[0]
