# emopylab 2026
"""LMEA (evolutionary algorithm for large-scale many-objective optimization).

Reference:
X. Zhang, Y. Tian, R. Cheng, and Y. Jin. A decision variable clustering based evolutionary algorithm
for large-scale many-objective optimization. IEEE Transactions on Evolutionary Computation, 2018,
22(1): 97-112.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, cosine_distance, crowding, decs, first_front, ga, ga_half, kmeans,
                                            nd_sort, objs, tournament)
from core.population import Population

ALGORITHM_FLAGS = {'LMEA': {'integer', 'large', 'multi', 'real'}}


def variable_clustering(algo, pop, nSel, nPer):
    """Classify decision variables into position (PV) and distance (DV) variables by probing the objective
    space with ``nPer`` perturbations of ``nSel`` sampled solutions per variable (charged to the budget)."""
    rng, N, D = algo.rng, len(pop), algo.D
    F = objs(pop)
    nd = first_front(F)
    fmin, fmax = F[nd].min(axis=0), F[nd].max(axis=0)
    if np.any(fmax == fmin):
        fmax, fmin = np.ones_like(fmax), np.zeros_like(fmin)
    angle, rmse = np.zeros((D, nSel)), np.zeros((D, nSel))
    sample = rng.integers(0, N, size=nSel)
    lo, up = algo.lower, algo.upper
    for i in range(D):
        X = np.tile(decs(pop[sample]), (nPer, 1))
        X[:, i] = rng.uniform(lo[i], up[i], size=len(X))
        new = objs(algo.evaluate(X))
        for j in range(nSel):
            P = (new[j::nSel] - fmin) / (fmax - fmin)
            P = P - P.mean(axis=0)
            v = np.linalg.svd(P, full_matrices=True)[2][0]
            v = v / np.linalg.norm(v)
            err = np.linalg.norm(P - (P @ v)[:, None] * v, axis=1)
            rmse[i, j] = np.sqrt(np.sum(err ** 2))
            cosine = abs(v.sum()) / np.linalg.norm(v) / np.linalg.norm(np.ones_like(v))
            angle[i, j] = np.degrees(np.arccos(min(1.0, cosine)))
    kind = np.mean(rmse, axis=1) < 1e-2
    result = kmeans(angle, 2, rng) + 1
    if np.any(result[kind] == 1) and np.any(result[kind] == 2):
        if np.mean(angle[(result == 1) & kind]) < np.mean(angle[(result == 2) & kind]):
            kind = kind & (result == 1)
        else:
            kind = kind & (result == 2)
    return np.where(~kind)[0], np.where(kind)[0]


def _cal_con(F):
    front_no, _ = nd_sort(F, None, np.inf)
    con = F.sum(axis=1)
    return front_no * (con.max() - con.min()) + con


def _correlation_analysis(algo, pop, DV, nCor):
    rng, lo, up = algo.rng, algo.lower, algo.upper
    sets = []
    for v in DV:
        related = []
        for d, grp in enumerate(sets):
            hit = False
            for u in grp:
                for _ in range(nCor):
                    p = pop[int(rng.integers(0, len(pop)))]
                    a2, b2 = rng.uniform(lo[v], up[v]), rng.uniform(lo[u], up[u])
                    X = np.tile(np.asarray(p.X, dtype=float), (3, 1))
                    X[0, v], X[1, u] = a2, b2
                    X[2, [v, u]] = [a2, b2]
                    F = objs(algo.evaluate(X))
                    if np.any((F[0] - np.asarray(p.F)) * (F[2] - F[1]) < 0):
                        related.append(d)
                        hit = True
                        break
                if hit:
                    break
        if not related:
            sets.append(np.array([v]))
        else:
            merged = np.concatenate([sets[d] for d in related] + [np.array([v])])
            sets = [s for d, s in enumerate(sets) if d not in related] + [merged]
    return sets


def _truncation(F, K, rng):
    with np.errstate(all="ignore"):
        Fn = np.nan_to_num((F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0)))
    cosine = 1.0 - cosine_distance(Fn)
    np.fill_diagonal(cosine, 0.0)
    choose = np.zeros(len(Fn), bool)
    choose[np.argmax(Fn, axis=0)] = True
    if choose.sum() > K:
        sel = np.where(choose)[0]
        choose = np.zeros(len(Fn), bool)
        choose[sel[rng.permutation(len(sel))[:K]]] = True
    else:
        while choose.sum() < K:
            un = np.where(~choose)[0]
            choose[un[int(np.argmin(np.max(cosine[np.ix_(~choose, choose)], axis=1)))]] = True
    return choose


class LMEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, nSel: int = 5, nPer: int = 50, nCor: int = 5, type: int = 1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.nSel, self.nPer, self.nCor, self.type = int(nSel), int(nPer), int(nCor), int(type)

    def start(self):
        self.PV, self.DV = variable_clustering(self, self.pop, self.nSel, self.nPer)
        self.DVSet = _correlation_analysis(self, self.pop, self.DV, self.nCor)

    def _convergence_optimization(self):
        rng, N, D = self.rng, len(self.pop), self.D
        con = _cal_con(objs(self.pop))
        for grp in self.DVSet:
            for _ in range(len(grp)):
                pool = tournament(2, 2 * N, con, rng=rng)
                off = decs(self.pop).copy()
                new = ga_half(self.problem, decs(self.pop[pool]), (1, 20, D / len(grp) / 2, 20), rng=rng)
                off[:, grp] = new[:, grp]
                child = self.evaluate(off)
                all_con = _cal_con(np.vstack([objs(self.pop), objs(child)]))
                con, new_con = all_con[:N], all_con[N:]
                upd = con > new_con
                self.pop[upd] = child[upd]
                con = np.where(upd, new_con, con)

    def _distribution_optimization(self):
        rng, N = self.rng, len(self.pop)
        off = decs(self.pop[tournament(2, N, _cal_con(objs(self.pop)), rng=rng)]).copy()
        new = ga(self.problem, decs(self.pop[rng.integers(0, N, size=N)]), rng=rng)
        off[:, self.PV] = new[:, self.PV]
        child = self.evaluate(off)
        merged = Population.merge(self.pop, child)
        F = objs(merged)
        front_no, max_f = nd_sort(F, None, N)
        nxt = front_no < max_f
        last = np.where(front_no == max_f)[0]
        nxt[last[_truncation(F[last], N - int(nxt.sum()), rng)]] = True
        self.pop = merged[nxt]

    def step(self):
        for _ in range(10):
            self._convergence_optimization()
        for _ in range(self.M):
            self._distribution_optimization()
