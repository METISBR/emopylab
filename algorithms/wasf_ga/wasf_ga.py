# emopylab 2026
"""WASF-GA (weighting achievement scalarizing function genetic algorithm).

Reference:
A. B. Ruiz, R. Saborido, and M. Luque. A preference-based evolutionary algorithm for multiobjective
optimization: the weighting achievement scalarizing function genetic algorithm. Journal of Global
Optimization, 2015, 62: 101-129.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'WASFGA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _fronts_class(V, F, point, ro):
    """Pick, round by round, the best solution of every weight vector (augmented ASF); returns pick order."""
    nv, N = len(V), len(F)
    remaining = list(range(N))
    order = []
    rounds = 0
    while len(order) < N:
        rounds += 1
        for i in range(min(nv, len(remaining))):
            sub = F[remaining] - point
            vals = np.max(sub * V[i], axis=1) + ro * np.sum(V[i] * sub, axis=1)
            j = int(np.where(vals == vals.min())[0][0])
            order.append(remaining.pop(j))
    return order, rounds


def _wasfga_sort(V, F, nsort, point, ro):
    order, rounds = _fronts_class(V, F, point, ro)
    nv = len(V)
    front = np.full(len(F), np.inf)
    for pos, idx in enumerate(order, start=1):
        it = int(np.ceil(pos / nv))
        front[idx] = np.inf if (it == 0 or it > nsort) else it
    return front, rounds


def _environmental_selection(V, pop, nsort, point, ro):
    F = objs(pop)
    front_no, max_f = _wasfga_sort(V, F, nsort, point, ro)
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    num = min(nsort - int(nxt.sum()), len(last))
    if num > 0:
        nxt[last[rank[:num]]] = True
    return pop[nxt], front_no[nxt], cd[nxt]


class WASFGA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, Point=None, ro: float = 0.0001, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Point = None if Point is None else np.asarray(Point, dtype=float)
        self.ro = float(ro)

    def start(self):
        n = len(self.pop)
        self.point = np.full(self.M, 0.5) if self.Point is None else self.Point
        if self.M == 2:
            eps, j = 0.001, np.arange(n)
            u = eps + j * (1 - 2 * eps) / (n - 1)
            self.V = np.column_stack([u, 1 - u])
        else:
            self.V, self.pop_size = uniform_point(self.pop_size, self.M)
        self.front_no, _ = _wasfga_sort(self.V, objs(self.pop), np.inf, self.point, self.ro)
        self.crowd = crowding(objs(self.pop), self.front_no)
        v = len(self.V)
        self.nsort = 2 if v >= n else n // v + 1

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.crowd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.front_no, self.crowd = _environmental_selection(
            self.V, Population.merge(self.pop, off), self.nsort, self.point, self.ro)
