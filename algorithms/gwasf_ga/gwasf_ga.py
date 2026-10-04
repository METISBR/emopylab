# emopylab 2026
"""GWASF-GA (global weighting achievement scalarizing function genetic algorithm).

Reference:
R. Saborido, A. B. Ruiz, and M. Luque. Global WASF-GA: An evolutionary algorithm in multiobjective
optimization to approximate the whole Pareto optimal front. Evolutionary computation, 2017, 25(2):
309-349.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'GWASFGA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _pick(F, remaining, ref, vec, ro):
    sub = F[remaining] - ref
    vals = np.max(sub * vec, axis=1) + ro * np.sum(vec * sub, axis=1)
    return int(np.where(vals == vals.min())[0][0])


def _sort(V, F, utop, nadir, nsort, ro):
    nv, N = len(V), len(F)
    remaining, order, rounds = list(range(N)), [], 0
    while len(order) < N:
        rounds += 1
        for i in range(nv // 2):
            order.append(remaining.pop(_pick(F, remaining, utop, V[2 * i], ro)))
            if len(order) == N:
                break
            order.append(remaining.pop(_pick(F, remaining, nadir, V[2 * i + 1], ro)))
            if len(order) == N:
                break
        if nv // 2 == 0:
            break
    front = np.full(N, np.inf)
    for pos, idx in enumerate(order, start=1):
        it = int(np.ceil(pos / nv))
        front[idx] = np.inf if (it == 0 or it > nsort) else it
    return front, rounds


def _environmental_selection(V, pop, utop, nadir, nsort, ro):
    F = objs(pop)
    front_no, max_f = _sort(V, F, utop, nadir, nsort, ro)
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    num = min(nsort - int(nxt.sum()), len(last))
    if num > 0:
        nxt[last[rank[:num]]] = True
    return pop[nxt], front_no[nxt], cd[nxt]


class GWASFGA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, ro: float = 0.0001, eps: float = 0.01, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.ro, self.eps = float(ro), float(eps)

    def start(self):
        n = len(self.pop)
        if self.M == 2:
            e, j = 0.001, np.arange(n)
            u = e + j * (1 - 2 * e) / (n - 1)
            self.V = np.column_stack([u, 1 - u])
        else:
            self.V, self.pop_size = uniform_point(self.pop_size, self.M)
        v = len(self.V)
        self.nsort = 2 if v >= n else n // v + 1
        F = objs(self.pop)
        self.nadir, self.utop = F.max(axis=0) + self.eps, F.min(axis=0) - self.eps
        self.front_no, _ = _sort(self.V, F, self.utop, self.nadir, self.nsort, self.ro)
        self.crowd = crowding(F, self.front_no)

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.crowd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.front_no, self.crowd = _environmental_selection(
            self.V, Population.merge(self.pop, off), self.utop, self.nadir, self.nsort, self.ro)
        P = objs(self.pop)
        self.utop = np.minimum(self.utop, P.min(axis=0) - self.eps)
        self.nadir = np.maximum(self.nadir, P.max(axis=0) + self.eps)
