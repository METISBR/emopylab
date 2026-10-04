# emopylab 2026
"""SGEA (steady-state and generational evolutionary algorithm).

Reference:
S. Jiang and S. Yang. A steady-state and generational evolutionary algorithm for dynamic
multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2017, 21(1): 65-82.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, first_front, ga_half, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'SGEA': {'binary', 'constrained', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _best(pop):
    C = cons(pop)
    mask = first_front(objs(pop), C if C.size else None)
    feas = mask & (np.all(C <= 0, axis=1) if C.size else True)
    return pop[feas] if np.any(feas) else pop[mask]


def _select(pop, N):
    fit = cal_fitness(objs(pop))
    nxt = fit < 1
    archive = pop[nxt]
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
        archive = pop[nxt]
    return pop[nxt], archive, fit[nxt]


class SGEA(LoopAlgorithm):
    """Steady-state and generational evolutionary algorithm for dynamic problems: environmental changes are
    detected by re-evaluating a tenth of the population and answered by re-evaluating the best half and moving the
    rest along the archive's shift."""

    def start(self):
        self.pop, self.archive, self.fit = _select(self.pop, len(self.pop))
        self.centroid = decs(self.archive).mean(axis=0)
        self.all_pop = []

    def _changed(self):
        pop, rng = self.pop, self.rng
        sel = rng.permutation(len(pop))[: int(np.ceil(len(pop) / 10))]
        old = pop[sel]
        new = self.evaluate(decs(old))
        return not (np.array_equal(objs(old), objs(new)) and np.array_equal(cons(old), cons(new)))

    def _change_response(self):
        pop, rng = self.pop, self.rng
        remain = ~_truncation(objs(pop), int(np.ceil(len(pop) / 2)))
        ridx, didx = np.where(remain)[0], np.where(~remain)[0]
        re = self.evaluate(decs(pop[ridx]))
        for k, ind in zip(ridx, re):
            pop[int(k)] = ind
        Ct = decs(self.archive).mean(axis=0)
        St = np.linalg.norm(Ct - self.centroid)
        arch = _best(pop[ridx])
        CA, CR = decs(arch).mean(axis=0), decs(pop[ridx]).mean(axis=0)
        X = decs(pop[didx])
        with np.errstate(all="ignore"):
            X = X + St * (CA - CR) / np.linalg.norm(CA - CR) + rng.standard_normal(X.shape) * St / 2 / np.sqrt(X.shape[1])
        new = self.evaluate(X)
        for k, ind in zip(didx, new):
            pop[int(k)] = ind
        self.archive = _best(pop)
        self.centroid = Ct

    def step(self):
        N, rng = self.N, self.rng
        if self._changed():
            self.all_pop.append(self.pop.copy(deep=False))
            self._change_response()
            self.fit = cal_fitness(objs(self.pop))
        elite = self.pop
        for _ in range(N):
            if rng.random() < 0.5:
                parents = decs(self.pop[tournament(2, 2, self.fit, rng=rng)])
            else:
                mp = tournament(2, 1, self.fit, rng=rng)
                parents = np.vstack([decs(self.pop[mp]), decs(self.archive[[int(rng.integers(len(self.archive)))]])])
            off = self.evaluate(ga_half(self.problem, parents, rng=rng))
            self.pop, self.archive, self.fit = _select(Population.merge(self.pop, off), N)
        self.pop, self.archive, self.fit = _select(Population.merge(self.pop, elite), N)
        if self.FE >= self.max_FE and self.all_pop:
            self.pop = Population.merge(*self.all_pop, self.pop)
