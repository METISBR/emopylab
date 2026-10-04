# emopylab 2026
"""MOEA-D-UR (mOEA/D with update when required).

Reference:
L. R. de Farias and A. F. Araujo. A decomposition-based many-objective evolutionary algorithm
updating weights when required. Swarm and Evolutionary Computation, 2022, 68: 100980.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, decs, first_front, ga, ga_half, kmeans, neighbors_of, objs,
                                            pdist2, unique_individuals)
from algorithms.moea_d_uraw.moea_d_uraw import uniformly_randomly_point
from core.population import Population

ALGORITHM_FLAGS = {'MOEADUR': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}

_FUN_THRESHOLD = [-1.989e-05, 0.0002034, 0.03376, 0.2373]
_FUN_RHO = [-0.4707, 0.8644, -0.1508, 0.05745]


def _update_weight(pop, W, Z, T, EP, nus):
    N, M = len(pop), pop[0].F.shape[0]
    dis = pdist2(objs(pop), objs(pop))
    np.fill_diagonal(dis, np.inf)
    dele = np.zeros(len(pop), bool)
    while dele.sum() < min(nus, len(EP)):
        remain = np.where(~dele)[0]
        sub = np.sort(dis[np.ix_(remain, remain)], axis=1)
        dele[remain[int(np.argmin(np.prod(sub[:, : min(M, len(remain))], axis=1)))]] = True
    pop, W = pop[~dele], W[~dele]
    comb = Population.merge(pop, EP)
    sel = np.zeros(len(comb), bool)
    sel[: len(pop)] = True
    dis = pdist2(objs(comb), objs(comb))
    np.fill_diagonal(dis, np.inf)
    while sel.sum() < min(N, len(sel)):
        sub = np.sort(dis[np.ix_(~sel, sel)], axis=1)
        best = int(np.argmax(np.prod(sub[:, : min(M, sub.shape[1])], axis=1)))
        sel[np.where(~sel)[0][best]] = True
    new_objs = objs(EP[sel[len(pop):]])
    with np.errstate(all="ignore"):
        temp = 1.0 / (new_objs - (Z - 0.001))
    temp[np.isinf(temp)] = 0.999999
    W = np.vstack([W, temp / temp.sum(axis=1, keepdims=True)])
    return comb[sel], W, neighbors_of(W, T)


def _best_per_weight(cands, W, Z):
    Fc = np.abs(objs(cands) - Z)
    g = np.stack([np.max(Fc * W[i], axis=1) for i in range(len(W))], axis=1)
    return cands[np.argmin(g, axis=0)]


class MOEADUR(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, start: float = 0.2, finish: float = 0.93, K: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.start_r, self.finish_r, self.K = float(start), float(finish), int(K)
        self.delta, self.nr, self.mini_generation = 0.9, 2, 1

    def initial_size(self):
        self.T = int(np.ceil(self.pop_size / 10))
        W, self.pop_size = uniformly_randomly_point(self.pop_size, self.M, self.rng)
        self.W = 1.0 / W / np.sum(1.0 / W, axis=1, keepdims=True)
        self.W_URP = self.W.copy()
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.Z = objs(self.pop).min(axis=0)
        self.EP = self.pop[first_front(objs(self.pop))]
        span = self.max_FE / self.N
        self.when_start, self.when_end = int(np.floor(self.start_r * span)), int(np.floor(self.finish_r * span))
        self.period, self.rho, self.nus_n, self.I_old = None, None, None, None

    def _space_divide(self):
        rng = self.rng
        idx = kmeans(objs(self.pop), self.K, rng)
        all_ind = Population.merge(self.pop, self.EP)
        for k in np.unique(idx):
            group = self.pop[idx == k]
            group_size = len(group)
            if len(unique_individuals(group)) > 2:
                Wg = self.W_URP[idx == k]
                wmax, wmin = Wg.max(axis=0), Wg.min(axis=0)
                Wn = self.W_URP * (wmax - wmin) + wmin
                cur = _best_per_weight(all_ind, Wn, self.Z)
                for _ in range(self.mini_generation):
                    parents = unique_individuals(cur)
                    pool = rng.integers(0, len(parents), size=group_size)
                    off = self.evaluate(ga(self.problem, decs(parents[pool]), rng=rng))
                    self.Z = np.minimum(self.Z, objs(off).min(axis=0))
                    cur = Population.merge(cur, off)
                all_ind = Population.merge(all_ind, cur)
        self._combine(all_ind)

    def _combine(self, cands):
        comb = unique_individuals(cands)
        if len(comb) < self.N:
            comb = cands
        cands2 = unique_individuals(_best_per_weight(comb, self.W, self.Z))
        for c in cands2:
            fc = np.asarray(c.F, dtype=float)
            g = np.max(np.abs((fc - self.Z) * self.W), axis=1)
            chosen = np.where(g == g.min())[0]
            if not all(self.pop[int(j)] is c for j in chosen):
                g_old = np.max(np.abs(objs(self.pop[chosen]) - self.Z) * self.W[chosen], axis=1)
                g_new = np.max(np.abs(fc - self.Z) * self.W[chosen], axis=1)
                self.pop[chosen[g_old >= g_new]] = c

    def step(self):
        rng, N, W = self.rng, self.N, self.W
        kids = []
        local = rng.random(N) < self.delta
        for i in range(N):
            P = self.B[i][rng.permutation(self.B.shape[1])] if local[i] else rng.permutation(N)
            child = self.evaluate(ga_half(self.problem, decs(self.pop[P[:2]]), rng=rng))
            kids.append(child[0])
            fo = objs(child)[0]
            self.Z = np.minimum(self.Z, fo)
            g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) * W[P], axis=1)
            g_new = np.max(np.abs(fo - self.Z) * W[P], axis=1)
            self.pop[P[np.where(g_old >= g_new)[0][: self.nr]]] = child[0]
        gen = self.FE / N
        if gen == self.when_start:
            X = np.unique(objs(self.pop), axis=0)
            X = X[first_front(X)]
            X = X / np.linalg.norm(X, axis=0)
            spreading = np.linalg.norm(X, 2) / 4
            threshold = np.polyval(_FUN_THRESHOLD, self.M)
            if spreading <= threshold:
                self.period, nus = 12, 0.25
            else:
                self.period, nus = 28, 0.075
            self.nus_n = int(round(nus * N))
            self.rho = np.polyval(_FUN_RHO, spreading)
            self.I_old = np.max(np.abs((objs(self.pop) - self.Z) * W), axis=1)
        if gen <= self.when_end:
            self.EP = Population.merge(self.EP, Population.create(kids))
        if self.when_start <= gen <= self.when_end and self.period is not None and gen % self.period == 0:
            I_new = np.max(np.abs((objs(self.pop) - self.Z) * W), axis=1)
            with np.errstate(all="ignore"):
                improvement = np.mean(1 - I_new / self.I_old)
            self.I_old = I_new
            if abs(improvement) <= self.rho:
                self.EP = unique_individuals(self.EP)
                self.EP = self.EP[first_front(objs(self.EP))]
                self.pop, self.W, self.B = _update_weight(self.pop, self.W, self.Z, self.T, self.EP, self.nus_n)
                self.I_old = np.max(np.abs((objs(self.pop) - self.Z) * self.W), axis=1)
                self._space_divide()
                self.EP = Population.create([])
