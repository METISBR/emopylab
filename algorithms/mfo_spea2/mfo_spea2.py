# emopylab 2026
"""MFO-SPEA2 (multiform optimization framework based on SPEA2).

Reference:
R. Jiao, B. Xue, and M. Zhang. A multiform optimization framework for constrained multiobjective
optimization. IEEE Transactions on Cybernetics, 2023, 53(8):5165-5177.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MFOSPEA2': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _con(pop):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), 1))


def reduce_boundary(eF, k, max_k, cp=10):
    z, near = 1e-8, 1e-15
    with np.errstate(all="ignore"):
        B = max_k / np.power(np.log((eF + z) / z), 1.0 / cp)
        B = np.where(B == 0, B + near, B)
        f = eF * np.exp(-((k / B) ** cp))
    f = np.where(np.abs(f - z) < near, z, f)
    eps = f - z
    eps[eps <= 0] = 0
    return eps


def _fitness(F, C):
    fit = cal_fitness(F, C)
    return fit, np.argsort(fit, kind="stable")           # "rank" is the sorting permutation, as in the reference


def env_selection(pop, N, C):
    fit, rank = _fitness(objs(pop), C)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt], rank[nxt]


class MFOSPEA2(LoopAlgorithm):
    """Multifactorial SPEA2 for constrained problems: a target population uses the true constraints while a source
    population uses constraints relaxed by an epsilon that shrinks over the run; both mate in one common pool."""

    def start(self):
        self.T = self.pop
        self.S = self.pop
        C = _con(self.T)
        self.iE = np.maximum(np.max(np.maximum(0, C), axis=0), 1)
        _, self.rT = _fitness(objs(self.T), C)
        _, self.rS = _fitness(objs(self.S), _con(self.S) - self.iE)

    def step(self):
        N, rng = self.N, self.rng
        eps = reduce_boundary(self.iE, int(np.ceil(self.FE / N)), int(np.ceil(self.max_FE / N)) - 1)
        pool = tournament(2, N, np.concatenate([self.rT, self.rS]).astype(float), rng=rng)
        P = Population.merge(self.T, self.S)
        off = self.evaluate(ga(self.problem, decs(P[pool]), rng=rng))
        Co = _con(off)
        self.T, self.rT = env_selection(Population.merge(self.T, off), N, np.vstack([_con(self.T), Co]))
        self.S, self.rS = env_selection(Population.merge(self.S, off), N, np.vstack([_con(self.S) - eps, Co - eps]))
        self.pop = self.T
