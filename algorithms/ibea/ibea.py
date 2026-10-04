# emopylab 2026
"""IBEA (indicator-based evolutionary algorithm).

Reference:
E. Zitzler and S. Kunzli. Indicator-based selection in multiobjective search. Proceedings of the
International Conference on Parallel Problem Solving from Nature, 2004, 832-842.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'IBEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def cal_fitness(F, kappa):
    """Binary additive-epsilon indicator fitness; returns ``(fitness, I, C)``."""
    N = len(F)
    with np.errstate(all="ignore"):
        Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    Fn = np.nan_to_num(Fn)
    I = np.max(Fn[:, None, :] - Fn[None, :, :], axis=2)                 # I[i, j] = max_m (f_i - f_j)
    C = np.max(np.abs(I), axis=0)
    with np.errstate(all="ignore"):
        fit = np.sum(-np.exp(-I / C[None, :] / kappa), axis=0) + 1.0
    return np.nan_to_num(fit, nan=1.0), I, C


def truncate_by_fitness(pop, N, kappa):
    nxt = list(range(len(pop)))
    fit, I, C = cal_fitness(objs(pop), kappa)
    while len(nxt) > N:
        x = int(np.argmin(fit[nxt]))
        with np.errstate(all="ignore"):
            fit = fit + np.exp(-I[nxt[x]] / C[nxt[x]] / kappa)
        del nxt[x]
    return pop[np.asarray(nxt, dtype=int)]


class IBEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, kappa: float = 0.05, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.kappa = float(kappa)

    def step(self):
        fit, _, _ = cal_fitness(objs(self.pop), self.kappa)
        pool = tournament(2, self.N, -fit, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = truncate_by_fitness(Population.merge(self.pop, off), self.N, self.kappa)
