# emopylab 2026
"""C-TSEA (constrained two-stage evolutionary algorithm).

Reference:
F. Ming, W. Gong, H. Zhen, S. Li, L. Wang, and Z. Liao. A simple two-stage evolutionary algorithm
for constrained multi-objective optimization. Knowledge-Based Systems, 2021, 228: 107263.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.armoea import last_selection, update_ref_point
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament, uniform_point
from algorithms.community_utils.spea import cal_fitness, overall_cv, truncation
from core.population import Population

ALGORITHM_FLAGS = {'CTSEA': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _spea_pick(pop, N):
    """SPEA-style selection on the constraint-aware fitness (mask over ``pop`` and the fitness)."""
    F = objs(pop)
    fit = cal_fitness(F, cons(pop))
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    return nxt, fit


def _update_archive(archive, pop, N):
    """Feasible solutions of ``pop`` join the archive; a short archive is completed with the least violating ones and
    a long one is thinned with the constraint-aware fitness."""
    feas = overall_cv(cons(pop)) == 0
    archive = Population.merge(archive, pop[feas]) if feas.any() else archive
    if len(archive) == N:
        return archive
    if len(archive) < N:
        inside = {id(i) for i in archive}
        rest = pop[np.array([id(i) not in inside for i in pop], dtype=bool)]
        order = np.argsort(overall_cv(cons(rest)), kind="stable")
        return Population.merge(archive, rest[order[: N - len(archive)]])
    nxt, _ = _spea_pick(archive, N)
    return archive[nxt]


class CTSEA(LoopAlgorithm):
    """Stage one ignores the constraints (indicator-based selection with adaptive reference points, while a feasible
    archive is collected); halfway through, the archive becomes the population and constraint-aware SPEA-style
    selection finishes the search."""

    def start(self):
        pop, N = self.pop, self.N
        self.W, _ = uniform_point(N, self.M)
        self.arm_archive, self.ref, self.range, _ = update_ref_point(objs(pop), self.W, None)
        self.archive = pop[overall_cv(cons(pop)) == 0]
        self.stage2 = False

    def _env_selection1(self, pop):
        N = self.N
        F = objs(pop)
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        last = np.where(front == maxf)[0]
        choose = last_selection(F[last], self.ref, self.range, N - int(nxt.sum()))
        nxt[last[choose]] = True
        pop = pop[nxt]
        self.range = self.range.copy()
        self.range[1] = objs(pop).max(axis=0)
        self.range[1, self.range[1] - self.range[0] < 1e-6] = 1
        return pop

    def step(self):
        rng, N = self.rng, self.N
        pop = self.pop
        if self.FE < 0.5 * self.max_FE:
            # the reference never ranks parents (all ties), so the tournament reduces to lower-index wins
            mate = tournament(2, len(pop), np.zeros(len(pop)), rng=rng)
            off = self.evaluate(ga(self.problem, decs(pop[mate]), rng=rng))
            self.arm_archive, self.ref, self.range, _ = update_ref_point(np.vstack([self.arm_archive, objs(off)]), self.W, self.range)
            self.archive = _update_archive(self.archive, Population.merge(pop, off), N)
            self.pop = self._env_selection1(Population.merge(pop, off))
            return
        if not self.stage2:
            pop, self.archive = self.archive, pop
            self.stage2 = True
        fit = cal_fitness(objs(pop), cons(pop))
        mate = tournament(2, N, fit, rng=rng)
        off = self.evaluate(ga(self.problem, decs(pop[mate]), rng=rng))
        allp = Population.merge(pop, off)
        nxt, fit = _spea_pick(allp, N)
        allp, fit = allp[nxt], fit[nxt]
        self.pop = allp[np.argsort(fit, kind="stable")]
