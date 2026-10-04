# emopylab 2026
"""C3M (constraint, multiobjective, multi-stage, multi-constraint evolutionary algorithm).

Reference:
R. Sun, J. Zou, Y. Liu, S. Yang, and J. Zheng. A multi-stage algorithm for solving multi-objective
optimization problems with multi-constraints. IEEE Transactions on Evolutionary Computation, 2023,
27(5): 1207-1219.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, de, ga, first_front, nd_sort, objs, tournament
from algorithms.community_utils.spea import cal_fitness, truncation
from core.population import Population

ALGORITHM_FLAGS = {'C3M': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _cal_fitness(pop, which):
    """Strength/density fitness where only constraint ``which`` (1-based) counts as violation (0 = none, > total = all)."""
    F = objs(pop)
    C = cons(pop)
    if which == "none":
        return cal_fitness(F)
    if which == "all":
        return cal_fitness(F, C)
    return cal_fitness(F, C[:, [which - 1]])


def _select(pop, N, processcon, totalcon):
    which = "none" if not processcon else ("all" if processcon > totalcon else processcon)
    fit = _cal_fitness(pop, which)
    F = objs(pop)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    pop, fit = pop[nxt], fit[nxt]
    order = np.argsort(fit, kind="stable")
    return pop[order], fit[order]


def _archive(pop, N):
    C = cons(pop)
    feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
    pop = pop[feas]
    if len(pop) == 0:
        return pop
    pop = pop[first_front(objs(pop))]
    if len(pop) > N:
        cd = crowding(objs(pop))
        pop = pop[np.argsort(-cd, kind="stable")[:N]]
    return pop


class C3M(LoopAlgorithm):
    """The constraints are handled one at a time: a population per constraint evolves in parallel, the order in which the
    constraints are then activated follows how hard each one is (rank of its population), a constraint is dropped once it is
    dominated by the others, and finally all populations are pooled with an archive under the full constraint set."""

    def __init__(self, pop_size: int = 100, type: int = 1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = int(type)

    def start(self):
        pop, N = self.pop, self.N
        self.total = cons(pop).shape[1]
        self.processcon = 0 if self.total else 1            # (an unconstrained problem has nothing to process)
        self.fit = _cal_fitness(pop, "none")
        self.arch = pop
        self.gen = 2
        self.pops = [[pop, _cal_fitness(pop, i + 1), i + 1, 0] for i in range(self.total)]
        self.obj_values = {1: float(np.sum(objs(pop)))}
        self.ns, self.flag, self.state = 0, True, False
        self.seq, self.index, self.processed = [], 0, []
        self.threshold = 1e-3

    def _all_pops(self):
        return Population.merge(*[p[0] for p in self.pops])

    def _is_stable(self):
        pop, N, M = self.pop, self.N, self.M
        front, _ = nd_sort(objs(pop), None, len(pop))
        nc = int(np.sum(front == 1))
        max_change = abs(self.obj_values[self.gen] - self.obj_values[self.gen - 1])
        if nc == N:
            thr = self.threshold * abs((self.obj_values[self.gen] / N) / M) * 10 ** (M - 2)
            return max_change <= thr
        return False

    def step(self):
        N, rng, total = self.N, self.rng, self.total
        pop = self.pop
        if self.processcon <= total and self.ns:
            self.ns = 0
            pop = self.evaluate(self.random_decs(N))
            self.fit = _cal_fitness(pop, self.processcon if self.processcon else "none")
        if self.flag and self.processcon > total:
            pool = Population.merge(self.arch, self._all_pops(), pop) if total else Population.merge(self.arch, pop)
            self.flag = False
            pop, self.fit = _select(pool, N, self.processcon, total)
        self.obj_values[self.gen] = float(np.sum(np.abs(objs(pop))))
        if self.type == 1:
            mate = tournament(2, 2 * N, self.fit, rng=rng)
            X = decs(pop)
            off = self.evaluate(de(self.problem, X, X[mate[: len(mate) // 2]], X[mate[len(mate) // 2:]], rng=rng))
        else:
            mate = tournament(2, N, self.fit, rng=rng)
            off = self.evaluate(ga(self.problem, decs(pop[mate]), rng=rng))
        if self.flag:
            for i in range(total):
                self.pops[i][0], self.pops[i][1] = _select(Population.merge(self.pops[i][0], off), N, i + 1, total)
        pop, self.fit = _select(Population.merge(pop, off), N, self.processcon, total)
        self.pop = pop
        if self.processcon <= total:
            self.state = self._is_stable()
        if self.state and self.processcon <= total:
            self.ns = 1
            all_pops = self._all_pops()
            front, _ = nd_sort(objs(all_pops), None, np.inf)
            if self.processcon == 0:
                ranks = [front[j * N: (j + 1) * N].min() for j in range(total)]
                self.seq = list(np.argsort(ranks, kind="stable")[::-1] + 1)
            else:
                self.processed.append(self.processcon)
                self.pops[self.processcon - 1][3] = 1
                minidx = front[(self.processcon - 1) * N: self.processcon * N].min()
                for i in range(total):
                    if i + 1 != self.processcon and front[i * N: (i + 1) * N].max() <= minidx:
                        self.pops[i][3] = 1
            if sum(p[3] for p in self.pops) < total:
                while self.pops[self.seq[self.index] - 1][3] == 1:
                    self.index += 1
                self.processcon = self.seq[self.index]
            else:
                self.processcon = total + 1
        if self.FE == self.max_FE * 0.7:
            self.processcon = total + 1
        if self.flag:
            self.arch = _archive(Population.merge(self.arch, pop, off), N)
        self.gen += 1
