# emopylab 2026
"""TSTI (two-stage evolutionary algorithm with three indicators).

Reference:
J. Dong, W. Gong, F. Ming, and L. Wang. A two-stage evolutionary algorithm based on three indicators
for constrained multi-objective optimization. Expert Systems with Applications, 2022, 195: 116499.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'TSTI': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def calculate_fcv(pop):
    C = np.maximum(cons(pop), 0)
    if C.size == 0:
        return np.zeros(len(pop))
    with np.errstate(all="ignore"):
        CV = C / C.max(axis=0)
    CV[:, np.isnan(CV[0])] = 0
    return CV.sum(axis=1) / C.shape[1]


def estimation(F, r):
    N = len(F)
    with np.errstate(all="ignore"):
        F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    fpr = F.sum(axis=1)
    d = np.sqrt(np.maximum(((F[:, None, :] - F[None, :, :]) ** 2).sum(axis=2), 0))
    np.fill_diagonal(d, np.inf)
    fi, fj = fpr[:, None], fpr[None, :]
    close = d < r
    case1, case2 = close & (fi <= fj), close & (fi > fj)
    sh = np.zeros((N, N))
    with np.errstate(all="ignore"):
        sh[case1] = (0.5 * (1 - d[case1] / r)) ** 2
        sh[case2] = (1.5 * (1 - d[case2] / r)) ** 2
    return fpr, np.sqrt(sh.sum(axis=1))


def _fit(pop, N, Epsilon, r):
    fpr, fcd = estimation(objs(pop), r)
    fcv = calculate_fcv(pop)
    fm, _ = nd_sort(np.column_stack([fpr, fcd]), None, N)
    obj = np.column_stack([fm + Epsilon * fcv, fcv])
    return obj, fcv


def _stage2_selection(pop, N):
    F, C = objs(pop), cons(pop)
    fit = cal_fitness(F, C if C.size else None)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(F[nxt], int(nxt.sum()) - N)]] = False
    p, f = pop[nxt], fit[nxt]
    r = np.argsort(f, kind="stable")
    return p[r], f[r]


class TSTI(LoopAlgorithm):
    """Two-stage constrained evolutionary algorithm with a tri-indicator ranking: stage I ranks by the
    non-dominated front of (progress, diversity) shifted by an epsilon-weighted constraint violation; stage II
    switches to the constrained SPEA2-style fitness."""

    def __init__(self, pop_size: int = 100, Epsilon0: float = 0.05, row: float = 1.01, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.eps0, self.row = float(Epsilon0), float(row)

    def start(self):
        N = self.N
        self.r = 1.0 / N ** (1.0 / self.M)
        self.eps = self.eps0
        self.PopObj, fcv = _fit(self.pop, N, self.eps, self.r)
        frank, _ = nd_sort(self.PopObj, None, N)
        self.fitness = frank + fcv / (fcv + 1)

    def step(self):
        N, rng = self.N, self.rng
        if self.FE <= 0.4 * self.max_FE:
            pool = tournament(2, N, self.fitness, rng=rng)
            off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=rng))
            OffObj, _ = _fit(off, N, self.eps, self.r)
            merged = Population.merge(self.pop, off)
            front, _ = nd_sort(np.vstack([self.PopObj, OffObj]), None, N)
            fcv = calculate_fcv(merged)
            fit = front + fcv / (fcv + 1)
            idx = np.argsort(fit, kind="stable")[:N]
            self.pop = merged[idx]
            self.PopObj, _ = _fit(self.pop, N, self.eps, self.r)
            self.eps *= self.row
        else:
            self.pop, fit2 = _stage2_selection(self.pop, N)
            pool = tournament(2, N, fit2, rng=rng)
            off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=rng))
            self.pop, _ = _stage2_selection(Population.merge(self.pop, off), N)
