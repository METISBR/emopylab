# emopylab 2026
"""IMTCMO (improved evolutionary multitasking-based CMOEA).

Reference:
K. Qiao, J. Liang, K. Yu, C. Yue, H. Lin, D. Zhang, and B. Qu. Evolutionary constrained
multiobjective optimization: scalable high-dimensional constraint benchmarks and algorithm. IEEE
Transactions on Evolutionary Computation, 2024, 28(4): 965-979.
"""

from __future__ import annotations

import numpy as np

from algorithms.apsea.apsea import _epsilon_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, objs
from algorithms.community_utils.eps_ea import de_pbest_1, de_rand_1, gn_r1r2r3
from algorithms.community_utils.spea import cal_fitness, overall_cv, truncation
from core.population import Population

ALGORITHM_FLAGS = {'IMTCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _main_selection(pop, N):
    F, C = objs(pop), cons(pop)
    fit = cal_fitness(F, C if C.size else None)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    pop, fit = pop[nxt], fit[nxt]
    o = np.argsort(fit, kind="stable")
    return pop[o], fit[o]


def _neighbour_pairing(rng, pop, zmin, nr=10):
    F = objs(pop) - zmin
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    cos = F @ F.T - 3 * np.eye(len(F))
    sind = np.argsort(-cos, axis=1, kind="stable")
    nb = sind[:, :nr]
    P = np.array([nb[i, rng.permutation(nr)[:2]] for i in range(len(F))])
    return pop, pop[P[:, 0]], pop[P[:, 1]]


class IMTCMO(LoopAlgorithm):
    """Two tasks share offspring: the main task solves the constrained problem, the auxiliary task an epsilon-relaxed version
    whose tolerance shrinks over the run.  Half of every task's offspring come from DE/rand/1 between neighbours in objective
    direction, half from DE with a p-best guide."""

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(self.pop_size))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop1 = infills
        self.fit1 = cal_fitness(objs(infills), cons(infills) if cons(infills).size else None)
        self.zmin1 = objs(infills).min(axis=0)
        self.pop2 = self.evaluate(self.random_decs(self.pop_size))
        self.fit2 = cal_fitness(objs(self.pop2), cons(self.pop2) if cons(self.pop2).size else None)
        self.zmin2 = objs(self.pop2).min(axis=0)
        cv = np.concatenate([overall_cv(cons(self.pop1)), overall_cv(cons(self.pop2))])
        self.var0 = float(cv.max()) if cv.max() > 0 else 1.0
        self.x = 0.0
        self.pop = self.pop1
        self._set_optimum()

    def _pbest_offspring(self, pop, fit, popsize, p=0.1):
        rng, N = self.rng, self.N
        X = decs(pop)
        perm = rng.permutation(N) + 1
        r1, r2, _ = gn_r1r2r3(rng, N, perm)
        arr = perm[:popsize] - 1
        best = np.argsort(fit, kind="stable")
        pnp = max(int(np.round(p * N)), 2)
        ri = np.maximum(1, np.ceil(rng.random(popsize) * pnp).astype(int))
        new = de_pbest_1(rng, self.lower, self.upper, X[arr], X[best[ri - 1]], X[r1[:popsize] - 1], X[r2[:popsize] - 1])
        return self.evaluate(new)

    def _task_offspring(self, pop, zmin, fit):
        rng, N = self.rng, self.N
        mating = pop[rng.permutation(N)]
        m1, m2, m3 = _neighbour_pairing(rng, mating, zmin)
        h = N // 2
        a = self.evaluate(de_rand_1(rng, self.lower, self.upper, decs(m1[:h]), decs(m2[:h]), decs(m3[:h])))
        b = self._pbest_offspring(pop, fit, h)
        return Population.merge(a, b)

    def step(self):
        N = self.N
        cp = (-np.log(self.var0) - 6) / np.log(1 - 0.5)
        var = self.var0 * (1 - self.x) ** cp
        off1 = self._task_offspring(self.pop1, self.zmin1, self.fit1)
        off2 = self._task_offspring(self.pop2, self.zmin2, self.fit2)
        self.zmin1 = np.minimum(self.zmin1, objs(off1).min(axis=0))
        self.zmin2 = np.minimum(self.zmin2, objs(off2).min(axis=0))
        self.pop1, self.fit1 = _main_selection(Population.merge(self.pop1, off1, off2), N)
        self.pop2, self.fit2 = _epsilon_selection(Population.merge(self.pop2, off2, off1), N, var)
        self.x += 1 / (self.max_FE / N)
        self.pop = self.pop1
