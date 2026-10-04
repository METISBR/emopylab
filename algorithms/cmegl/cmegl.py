# emopylab 2026
"""CMEGL (constrained evolutionary multitasking with global and local auxiliary tasks).

Reference:
K. Qiao, J. Liang, Z. Liu, K. Yu, C. Yue, and B. Qu. Evolutionary multitasking with global and local
auxiliary tasks for constrained multi-objective optimization. IEEE/CAA Journal of Automatica Sinica,
2023, 10(10): 1951-1964.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness, environmental_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, ga_half, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'CMEGL': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _cv(pop):
    C = cons(pop)
    return np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))


def _mean_positive(pop):
    cv = _cv(pop)
    pos = cv[cv > 0]
    return float(pos.mean()) if len(pos) else 0.0


def env_selection_lat(pop, N, VAR):
    """Local auxiliary task: keep the solutions whose violation does not exceed VAR, ranked with the violation as an
    additional objective."""
    cv = _cv(pop)
    f = pop[cv <= VAR]
    if len(f) == 0:
        return None, np.zeros(0)
    fcv = _cv(f)
    fit = cal_fitness(np.column_stack([objs(f), fcv]))
    if len(f) <= N:
        r = np.argsort(fit, kind="stable")
        return f[r], fit[r]
    nxt = fit < 1
    if nxt.sum() <= N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    else:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(f)[nxt], int(nxt.sum()) - N)]] = False
    p, ft = f[nxt], fit[nxt]
    r = np.argsort(ft, kind="stable")
    return p[r], ft[r]


class CMEGL(LoopAlgorithm):
    """Constrained multi-objective EA with a global and a local auxiliary task: the main population keeps the
    constraints, a global helper ignores them until its objective spread settles, and a local helper keeps only the
    solutions below the current violation level of the offspring."""

    def start(self):
        N = self.N
        self.P1 = self.pop
        self.f1 = cal_fitness(objs(self.P1), cons(self.P1))
        self.P2 = self.evaluate(self.random_decs(N))
        self.f2 = cal_fitness(objs(self.P2))
        self.P3 = self.evaluate(self.random_decs(N))
        self.f3 = cal_fitness(objs(self.P3), cons(self.P3))
        self.VAR0 = _mean_positive(self.P1)
        self.cnt, self.flag, self.std_obj = 0, 0, {}

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        self.cnt += 1
        if self.flag == 0:
            self.std_obj[self.cnt] = objs(self.P2).std(axis=0, ddof=1)
            if self.cnt > 100:
                block = np.array([self.std_obj[k] for k in range(self.cnt - 100, self.cnt + 1)])
                if np.sum(block.std(axis=0, ddof=1) < 0.5) == self.M:
                    self.flag = 1
        off1 = self.evaluate(ga_half(pr, decs(self.P1[tournament(2, N, self.f1, rng=rng)]), rng=rng))
        off2 = None
        if self.flag == 0:
            off2 = self.evaluate(ga_half(pr, decs(self.P2[tournament(2, N, self.f2, rng=rng)]), rng=rng))
        off3 = None
        if self.P3 is not None and len(self.P3) > 1:
            k = min(len(self.P3), N // 2)
            off3 = self.evaluate(ga(pr, decs(self.P3[tournament(2, k, self.f3, rng=rng)]), rng=rng))
        self.P1, self.f1 = environmental_selection(Population.merge(self.P1, off2, off3), N, True)
        self.P1, self.f1 = environmental_selection(Population.merge(self.P1, off1), N, True)
        if self.flag == 0:
            self.P2, self.f2 = environmental_selection(Population.merge(self.P2, off1, off2, off3), N, False)
        pool = Population.merge(*[p for p in (self.P3, off1, off2, off3) if p is not None])
        self.P3, self.f3 = env_selection_lat(pool, N, self.VAR0)
        self.VAR0 = _mean_positive(off1)
        self.pop = self.P1
