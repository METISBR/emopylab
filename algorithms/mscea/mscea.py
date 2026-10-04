# emopylab 2026
"""MSCEA (multi-stage constrained multi-objective evolutionary algorithm).

Reference:
Y. Zhang, Y. Tian, H. Jiang, X. Zhang, and Y. Jin. Design and analysis of helper-problem-assisted
evolutionary algorithm for constrained multiobjective optimization. Information Sciences, 2023, 648:
119547.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation
from algorithms.cmoea_ms.cmoea_ms import cal_fitness, cal_sde
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MSCEA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _con(pop):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), 1))


def _cv(pop):
    return np.sum(np.maximum(0, _con(pop)), axis=1)


def reduce_boundary(eF, k, max_k, cp):
    z, near = 1e-8, 1e-15
    with np.errstate(all="ignore"):
        B = max_k / np.power(np.log((eF + z) / z), 1.0 / cp)
        B = np.where(B == 0, B + near, B)
        f = eF * np.exp(-((k / B) ** cp))
    f = np.where(np.abs(f - z) < near, z, f)
    eps = f - z
    eps[eps <= 0] = 0
    return eps


def _trim(pop, nxt, N):
    if nxt.sum() < N:
        return None
    return nxt


def _survive(pop, fit, N):
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return nxt


def archive(pop, N):
    C = cons(pop)
    pop = pop[np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)]
    if len(pop) == 0:
        return pop
    fit = cal_fitness(objs(pop))
    nxt = fit < 1
    if nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt]


def env_selection1(pop, N, eps):
    C = np.maximum(0, _con(pop))
    ok = np.sum(C <= eps, axis=1) == C.shape[1]
    if ok.sum() > N:
        pop = pop[ok]
        fit = cal_fitness(objs(pop), _cv(pop))
        return pop[_survive(pop, fit, N)]
    return pop[np.argsort(_cv(pop), kind="stable")[:N]]


def env_selection2(pop, N, eps):
    C = np.maximum(0, _con(pop))
    ok = np.sum(C <= eps, axis=1) == C.shape[1]
    if ok.sum() > N:
        pop = pop[ok]
        fit = cal_fitness(np.column_stack([objs(pop), _cv(pop)]))
    else:
        fit = cal_fitness(np.column_stack([cal_sde(objs(pop)), _cv(pop)]))
    nxt = _survive(pop, fit, N)
    return pop[nxt], fit[nxt]


class MSCEA(LoopAlgorithm):
    """Multi-stage constraint-handling co-evolution: the main population tightens an epsilon threshold once it is
    entirely inside it, the helper population reacts to growth of its maximum violation, and an archive of
    feasible non-dominated solutions is returned at the end."""

    def __init__(self, pop_size: int = 100, cp: float = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.cp = float(cp)

    def start(self):
        N = self.N
        self.P1 = self.pop
        self.P2 = self.evaluate(self.random_decs(N))
        self.nCon = _con(self.P1).shape[1]
        e1 = np.max(np.maximum(0, _con(self.P1)), axis=0)
        e2 = np.max(np.maximum(0, _con(self.P2)), axis=0)
        e1[e1 == 0] = 1
        e2[e2 == 0] = 1
        self.iE1, self.iE2 = e1, e2
        self.eps1, self.eps2 = e1.copy(), e2.copy()
        cv2 = _cv(self.P2)
        self.fit2 = cal_fitness(np.column_stack([cal_sde(objs(self.P2)), cv2]))
        self.max_cv2 = {1: float(cv2.max())}
        self.asc2 = 0
        self.arch = archive(Population.merge(self.P1, self.P2), N)

    def step(self):
        N, rng, cp, pr = self.N, self.rng, self.cp, self.problem
        G = int(np.ceil(self.max_FE / N))
        gen = int(np.ceil(self.FE / N))
        pc1, pc2 = np.maximum(0, _con(self.P1)), np.maximum(0, _con(self.P2))
        if np.sum(np.sum(pc1 <= self.eps1, axis=1) == self.nCon) == len(self.P1):
            self.eps1 = reduce_boundary(self.iE1, gen, G - 1, cp)
        cv2 = pc2.sum(axis=1)
        self.max_cv2[gen] = float(cv2.max())
        prev = self.max_cv2.get(int(np.ceil((self.FE - N) / N)), self.max_cv2[gen])
        if self.max_cv2[gen] - prev > 0:
            self.asc2 += 1
            self.eps2 = reduce_boundary(self.iE2, gen - self.asc2, G - 1, cp)
        elif np.sum(np.sum(pc2 <= self.eps2, axis=1) == self.nCon) == len(self.P2):
            self.asc2 = 0
            self.eps2 = reduce_boundary(self.iE2, gen, G - 1, cp)
        m1 = tournament(2, N, np.sum(np.maximum(0, _con(self.P1) - self.eps1), axis=1), rng=rng)
        m2 = tournament(2, N, self.fit2, rng=rng)
        off1 = self.evaluate(ga_half(pr, decs(self.P1[m1]), rng=rng))
        off2 = self.evaluate(ga_half(pr, decs(self.P2[m2]), rng=rng))
        self.P1 = env_selection1(Population.merge(self.P1, off1, off2), N, self.eps1)
        self.P2, self.fit2 = env_selection2(Population.merge(self.P2, off1, off2), N, self.eps2)
        arch = Population.merge(self.arch, self.P1, self.P2)
        _, uni = np.unique(objs(arch), axis=0, return_index=True)
        self.arch = archive(arch[uni], N)
        self.pop = self.P1
        if self.FE >= self.max_FE and len(self.arch):
            self.pop = self.arch
