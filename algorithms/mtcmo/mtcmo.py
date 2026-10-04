# emopylab 2026
"""MTCMO (multitasking constrained multi-objective optimization).

Reference:
K. Qiao, K. Yu, B. Qu, J. Liang, H. Song, C. Yue, H. Lin, and K. C. Tan. Dynamic auxiliary task-
based evolutionary multitasking for constrained multi-objective optimization. IEEE Transactions on
Evolutionary Computation, 2023, 27(3): 642-656.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness, environmental_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MTCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _cvsum(pop):
    C = cons(pop)
    return np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))


def _pick(pop, fit, K):
    """Keep the ``fit < 1`` set, filling up to K by rank or truncating down to K; returns the mask."""
    nxt = fit < 1
    if nxt.sum() <= K:
        nxt[np.argsort(fit, kind="stable")[:K]] = True
    else:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - K)]] = False
    return nxt


def _sorted(pop, fit, mask, shift=0.0):
    pop, fit = pop[mask], fit[mask] + shift
    r = np.argsort(fit, kind="stable")
    return pop[r], fit[r]


def auxiliary_selection(pop, N, VAR):
    cv = _cvsum(pop)
    fi, ii = np.where(cv <= VAR)[0], np.where(cv > VAR)[0]
    fpop, ipop = pop[fi], pop[ii]
    if len(fi) == 0:
        fit = cal_fitness(objs(ipop), cons(ipop))
        return _sorted(ipop, fit, _pick(ipop, fit, N))
    if len(fi) <= N:
        ffit = cal_fitness(np.column_stack([objs(fpop), cv[fi]]))
        nxt = ffit < 1
        nxt[np.argsort(ffit, kind="stable")[: len(fi)]] = True
        fpop, ffit = _sorted(fpop, ffit, nxt)
        if len(ii) == 0:
            return fpop, ffit
        R = N - len(fpop)
        ifit = cal_fitness(objs(ipop), cons(ipop))
        ipop, ifit = _sorted(ipop, ifit, _pick(ipop, ifit, R), shift=ffit.max())
        return Population.merge(fpop, ipop), np.concatenate([ffit, ifit])
    ffit = cal_fitness(np.column_stack([objs(fpop), cv[fi]]))
    return _sorted(fpop, ffit, _pick(fpop, ffit, N))


class MTCMO(LoopAlgorithm):
    """Multitasking constrained MOEA: the main task keeps the constraints, the auxiliary task tolerates violations
    up to a threshold that decays along the run; both exchange offspring."""

    def start(self):
        N = self.N
        self.P1 = self.pop
        self.P2 = self.evaluate(self.random_decs(N))
        self.f1 = cal_fitness(objs(self.P1), cons(self.P1))
        self.f2 = cal_fitness(objs(self.P2), cons(self.P2))
        C = np.vstack([c for c in (cons(self.P1), cons(self.P2)) if c.size]) if (cons(self.P1).size) else np.zeros((0, 0))
        v0 = float(np.max(np.sum(np.maximum(C, 0), axis=1))) if C.size else 0.0
        self.VAR0 = v0 if v0 != 0 else 1.0
        self.X = 0.0

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        cp = (-np.log(self.VAR0) - 6) / np.log(1 - 0.5)
        VAR = self.VAR0 * max(1 - self.X, 0.0) ** cp
        off1 = self.evaluate(ga_half(pr, decs(self.P1[tournament(2, N, self.f1, rng=rng)]), rng=rng))
        off2 = self.evaluate(ga_half(pr, decs(self.P2[tournament(2, N, self.f2, rng=rng)]), rng=rng))
        self.P1, self.f1 = environmental_selection(Population.merge(self.P1, off1, off2), N, True)
        self.P2, self.f2 = auxiliary_selection(Population.merge(self.P2, off2, off1), N, VAR)
        self.X += 1 / (self.max_FE / N)
        self.pop = self.P1
