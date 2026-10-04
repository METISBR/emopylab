# emopylab 2026
"""HEA (hyper-dominance based evolutionary algorithm).

Reference:
Z. Liu, F. Han, Q. Ling, H. Han, and J. Jiang. A many-objective optimization evolutionary algorithm
based on hyper-dominance degree. Swarm and Evolutionary Computation, 2023, 83: 101411.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, ga, objs, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'HEA': {'binary', 'many', 'multi', 'permutation', 'real'}}


def _domination_cal(F, zmin, zmax, T):
    n = len(F)
    Fn = F / (zmax - zmin)
    err = Fn[:, None, :] - Fn[None, :, :]                       # err[i, j] = f_i - f_j
    max_err, min_err = err.max(axis=2), err.min(axis=2)
    with np.errstate(all="ignore"):
        H = -min_err / max_err
    same = np.all(err == 0, axis=2)
    dominated = np.zeros(n, bool)
    hd = np.zeros(n)
    for i in range(n):
        eq = np.zeros(n, bool)
        eq[i + 1:] = same[i, i + 1:]
        dominated[eq] = True
        h = H[i].copy()
        h[eq] = -np.inf
        h[max_err[i] <= 0] = np.inf
        hd[i] = h.min()
    dominated[hd < T] = True
    return ~dominated, hd


def _environmental_selection(pop, hd, zmin, zmax, N, W):
    F = objs(pop)
    zmin = np.minimum(zmin, F.min(axis=0))
    nf = np.maximum(zmax - zmin, 1e-6)
    Fn = (F - zmin) / nf
    idx = np.argmin(cosine_distance(Fn, W), axis=0)
    chosen = np.zeros(len(F), bool)
    chosen[idx[:N]] = True
    new_pop, new_hd = pop[chosen], hd[chosen]
    hd = hd.copy()
    for _ in range(N - int(chosen.sum())):
        j = int(np.argmax(hd))
        new_pop = Population.merge(new_pop, pop[j:j + 1])
        new_hd = np.append(new_hd, hd[j])
        hd[j] = -np.inf
    return new_pop, new_hd


class HEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, MaxT: float = 0.05, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.MaxT = float(MaxT)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        self.T = 0.0
        self.step_T = self.MaxT / (self.max_FE / self.N)
        self.work = self.pop
        self.solutions = self.pop
        self.zmin = F.min(axis=0)
        self.zmax = np.maximum(F.max(axis=0), self.zmin + 1e-6)
        self.pop = self.solutions

    def step(self):
        rng, N = self.rng, self.N
        perm = rng.permutation(len(self.work))
        off = self.evaluate(ga(self.problem, decs(self.work[perm]), rng=rng))
        pop = Population.merge(off, self.solutions)
        self.zmin = np.minimum(self.zmin, objs(pop).min(axis=0))
        nd, hd = _domination_cal(objs(pop), self.zmin, self.zmax, self.T)
        pop, hd = pop[nd], hd[nd]
        self.zmax = np.maximum(objs(pop).max(axis=0), self.zmin + 1e-6)
        self.T += self.step_T
        self.solutions, hd = _environmental_selection(pop, hd, self.zmin, self.zmax, N, self.W)
        work = self.solutions.copy(deep=False)
        r = rng.integers(0, len(self.solutions), size=len(self.solutions))
        for i in range(len(self.solutions)):
            if hd[i] < hd[r[i]]:
                work[i] = self.solutions[r[i]]
        self.work = work
        self.pop = self.solutions
