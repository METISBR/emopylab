# emopylab 2026
"""CMME (constrained many-objective evolutionary algorithm with enhanced mating and environmental selections).

Reference:
F. Ming, W. Gong, L. Wang, and L. Gao. A constrained many-objective optimization evolutionary
algorithm with enhanced mating and environmental selections. IEEE Transactions on Cybernetics, 2023,
53(8): 4934-4946.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament, uniform_point
from algorithms.community_utils.spea import cal_fitness
from core.population import Population

ALGORITHM_FLAGS = {'CMME': {'binary', 'constrained', 'integer', 'label', 'many', 'permutation', 'real'}}


def _regions(F, W):
    with np.errstate(all="ignore"):
        Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        cos = (Fn @ W.T) / (np.linalg.norm(Fn, axis=1)[:, None] * np.linalg.norm(W, axis=1)[None, :])
    return np.argmax(np.where(np.isnan(cos), -np.inf, cos), axis=1)


def _density(F, W):
    region = _regions(F, W)
    counter = np.bincount(region, minlength=len(W))
    return counter[region].astype(float)


def _normalise(F, z, znad):
    N, M = F.shape
    z = np.minimum(z, F.min(axis=0))
    Wm = np.zeros((M, M)) + 1e-6
    np.fill_diagonal(Wm, 1)
    with np.errstate(all="ignore"):
        ASF = np.column_stack([np.max(np.abs((F - z) / (znad - z)) / Wm[i], axis=1) for i in range(M)])
    extreme = np.argmin(ASF, axis=0)
    try:
        hyper = np.linalg.solve(F[extreme] - z, np.ones(M))
        a = 1.0 / hyper + z
    except np.linalg.LinAlgError:
        a = np.full(M, np.nan)
    if np.any(np.isnan(a)) or np.any(a <= z):
        a = F.max(axis=0)
    with np.errstate(all="ignore"):
        return (F - z) / (a - z)


def _t_nd_sort(F, W):
    """Rank inside every subregion by the (theta-)penalised boundary distance (a total order per region)."""
    N, NW = len(F), len(W)
    Fn = _normalise(F, F.min(axis=0), F.max(axis=0))
    normp = np.linalg.norm(Fn, axis=1)
    with np.errstate(all="ignore"):
        cos = (Fn @ W.T) / (normp[:, None] * np.linalg.norm(W, axis=1)[None, :])
        d1 = normp[:, None] * cos
        d2 = normp[:, None] * np.sqrt(np.maximum(1 - cos ** 2, 0))
    cls = np.argmin(np.where(np.isnan(d2), np.inf, d2), axis=1)
    theta = np.zeros(NW) + 5
    theta[np.sum(W > 1e-4, axis=1) == 1] = 1e6
    rank = np.zeros(N)
    for i in range(NW):
        c = np.where(cls == i)[0]
        order = np.argsort(d1[c, i] + theta[i] * d2[c, i], kind="stable")
        rank[c[order]] = np.arange(1, len(c) + 1)
    return rank


def _environmental_selection(pop, W, N):
    F, C = objs(pop), cons(pop)
    fit = cal_fitness(F, C if C.size else None)
    sc_mask = fit < 1
    Sc = np.where(sc_mask)[0]
    if len(Sc) == N:
        return pop[Sc]
    if len(Sc) > N:
        Fs = F[Sc]
        front, maxno = nd_sort(Fs, None, np.inf)
        if maxno == 1:
            front = _t_nd_sort(Fs, W)
            maxno = int(front.max())
        S = []
        for i in range(1, maxno + 1):
            S.extend(Sc[front == i])
            if len(S) >= N:
                break
        S = list(S)
        while len(S) > N:
            Fsub = F[S]
            region = _regions(Fsub, W)
            counter = np.bincount(region, minlength=len(W))
            crowded = int(np.argmax(counter))
            members = np.where(region == crowded)[0]
            dist = np.sqrt(np.maximum(np.sum(Fsub[members] ** 2, 1)[:, None] + np.sum(Fsub[members] ** 2, 1)[None, :] - 2 * Fsub[members] @ Fsub[members].T, 0))
            dist = np.where(dist == 0, np.inf, dist)
            rows = np.where(np.any(dist == dist.min(), axis=1))[0] if np.isfinite(dist.min()) else np.arange(len(members))
            St = members[rows]
            reg_st = _regions(Fsub[St], W)
            z = Fsub[St].min(axis=0)
            g = np.max(np.abs(Fsub[St] - z) / W[reg_st], axis=1)
            del S[int(St[int(np.argmax(g))])]
        return pop[np.array(S)]
    SI = np.where(~sc_mask)[0]
    f1 = np.sum(np.maximum(0, C[SI]), axis=1) if C.size else np.zeros(len(SI))
    reg = _regions(F[SI], W)
    z = F[SI].min(axis=0)
    f2 = np.max(np.abs(F[SI] - z) / W[reg], axis=1)
    front, maxno = nd_sort(np.column_stack([f1, f2]), None, np.inf)
    S = list(Sc)
    last = 0
    for i in range(1, int(maxno) + 1):
        S.extend(SI[front == i])
        if len(S) >= N:
            last = i
            break
    fl = SI[front == last]
    delete_n = len(S) - N
    cv = np.sum(np.maximum(0, C[fl]), axis=1) if C.size else np.zeros(len(fl))
    worst = set(fl[np.argsort(-cv, kind="stable")[:delete_n]].tolist())
    return pop[np.array([s for s in S if s not in worst])]


class CMME(LoopAlgorithm):
    """Mating favours individuals in sparsely populated subregions (constraint-aware fitness is zero for all solutions in
    this formulation); survival keeps constrained non-dominated solutions, thinning crowded subregions through a Tchebycheff
    distance, or, if too few are feasible, the least violating / best-converged infeasible ones."""

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def step(self):
        rng, N = self.rng, self.N
        pop = self.pop
        density = _density(objs(pop), self.W)
        fitness = np.zeros(len(pop))
        mate = tournament(2, N, fitness, density, rng=rng)
        off = self.evaluate(ga(self.problem, decs(pop[mate]), rng=rng))
        self.pop = _environmental_selection(Population.merge(pop, off), self.W, N)
