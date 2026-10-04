# emopylab 2026
"""CAEAD (dual-population evolutionary algorithm based on alternative evolution and degeneration).

Reference:
J. Zou, R. Sun, S. Yang, and J. Zheng. A dual-population algorithm based on alternative evolution
and degeneration for solving constrained multi- objective optimization problems. Informaction
Scinece, 2021, 239: 89-102.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, ga_half, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'CAEAD': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _overall_cv(c):
    return np.sum(np.abs(np.maximum(c, 0)), axis=1) if np.size(c) else np.zeros(len(c))


def _select(pop, N, epsilon, use_eps=True):
    """Fitness-based survival where violations up to ``epsilon`` are ignored (epsilon=0: strict feasibility)."""
    F = objs(pop)
    fit = cal_fitness(F, cons(pop), epsilon)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(F[nxt], int(nxt.sum()) - N)]] = False
    pop, fit = pop[nxt], fit[nxt]
    rank = np.argsort(fit, kind="stable")
    return pop[rank], fit[rank]


class CAEAD(LoopAlgorithm):
    """Co-evolutionary algorithm with an epsilon-relaxed helper population: the helper stage starts after its
    objective sum settles, then epsilon shrinks geometrically and restarts from the largest value seen."""

    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type

    def start(self):
        self.pop1 = self.pop
        self.pop2 = self.evaluate(self.random_decs(self.N))
        self.fit1 = cal_fitness(objs(self.pop1), cons(self.pop1), 0)
        self.fit2 = cal_fitness(objs(self.pop2), cons(self.pop2), 1e6)
        self.min_eps, self.change_thr = 1e-4, 1e-2
        self.max_change, self.eps_k, self.tao, self.max_ep = 1.0, 1e8, 0.05, 0.0
        self.gen, self.stage, self.obj_values = 1, False, []

    def step(self):
        N, rng, P1, P2, pr = self.N, self.rng, self.pop1, self.pop2, self.problem
        cv2 = _overall_cv(cons(P2))
        self.obj_values.append(float(objs(P2).sum()))
        if self.type == 1:
            m1, m2 = tournament(2, 2 * N, self.fit1, rng=rng), tournament(2, 2 * N, self.fit2, rng=rng)
            off1 = self.evaluate(de(pr, decs(P1), decs(P1[m1[:N]]), decs(P1[m1[N:]]), rng=rng))
            off2 = self.evaluate(de(pr, decs(P2), decs(P2[m2[:N]]), decs(P2[m2[N:]]), rng=rng))
        else:
            m1, m2 = tournament(2, N, self.fit1, rng=rng), tournament(2, N, self.fit2, rng=rng)
            off1 = self.evaluate(ga_half(pr, decs(P1[m1]), rng=rng))
            off2 = self.evaluate(ga_half(pr, decs(P2[m2]), rng=rng))
        front2, _ = nd_sort(objs(P2), None, len(P2))
        NC2 = int(np.sum(front2 == 1))
        if self.gen != 1:
            self.max_change = abs(self.obj_values[-1] - self.obj_values[-2])
        if self.max_change <= self.change_thr and NC2 == N and not self.stage:
            self.eps_k = float(cv2.max())
            self.stage = True
        off3 = None
        if self.stage:
            if self.type == 1:
                off3 = self.evaluate(de(pr, decs(P1), decs(P2[m2[:N]]), decs(P2[m2[N:]]), rng=rng))
            else:
                parts = [self.evaluate(ga_half(pr, np.vstack([decs(P1[[m1[i]]]), decs(P2[[m2[i]]])]), rng=rng)) for i in range(N // 2)]
                off3 = Population.merge(*parts) if len(parts) > 1 else parts[0]
            if self.eps_k > self.min_eps:
                self.eps_k, self.stage = (1 - self.tao) * self.eps_k, True
            else:
                self.eps_k, self.stage = self.max_ep, False
        if self.eps_k < 9e5:
            self.max_ep = max(self.max_ep, self.eps_k)
        self.pop1, self.fit1 = _select(Population.merge(P1, off1, off2, off3), N, 0)
        self.pop2, self.fit2 = _select(Population.merge(P2, off2), N, self.eps_k)
        self.pop = self.pop1
        self.gen += 1
