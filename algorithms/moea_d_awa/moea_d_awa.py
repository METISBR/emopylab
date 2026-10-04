# emopylab 2026
"""MOEA-D-AWA (mOEA/D with adaptive weight adjustment).

Reference:
Y. Qi, X. Ma, F. Liu, L. Jiao, J. Sun, and J. Wu. MOEA/D with adaptive weight adjustment.
Evolutionary Computation, 2014, 22(2): 231-264.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, first_front, ga_half, decs, neighbors_of, objs, pdist2, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MOEADAWA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _prune_by_product(F, n_remove, M):
    """Repeatedly drop the point with the smallest product of its M smallest neighbour distances."""
    dis = pdist2(F, F)
    np.fill_diagonal(dis, np.inf)
    dele = np.zeros(len(F), bool)
    while dele.sum() < n_remove:
        remain = np.where(~dele)[0]
        sub = np.sort(dis[np.ix_(remain, remain)], axis=1)
        worst = int(np.argmin(np.prod(sub[:, : min(M, len(remain))], axis=1)))
        dele[remain[worst]] = True
    return dele


def update_ep(EP, off, nEP):
    EP = Population.merge(EP, off)
    EP = EP[first_front(objs(EP))]
    M = EP[0].F.shape[0]
    dele = _prune_by_product(objs(EP), len(EP) - nEP, M)
    return EP[~dele]


def _update_weight(pop, W, Z, EP, nus):
    N, M = len(pop), pop[0].F.shape[0]
    comb = Population.merge(pop, EP)
    Fc = np.abs(objs(comb) - Z)
    g = np.stack([np.max(Fc * W[i], axis=1) for i in range(len(W))], axis=1)
    pop = comb[np.argmin(g, axis=0)]
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
        temp = 1.0 / (new_objs - Z)
    W = np.vstack([W, temp / temp.sum(axis=1, keepdims=True)])
    return comb[sel], W


class MOEADAWA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, rate_update_weight: float = 0.05, rate_evol: float = 0.8, wag: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.rate_update_weight, self.rate_evol, self.wag = float(rate_update_weight), float(rate_evol), float(wag)

    def initial_size(self):
        W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.W = 1.0 / W / np.sum(1.0 / W, axis=1, keepdims=True)
        self.T = int(np.ceil(self.pop_size / 10))
        self.nr = int(np.ceil(self.pop_size / 100))
        self.nEP = int(np.ceil(self.pop_size * 1.5))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        self.Z = F.min(axis=0)
        self.Pi = np.ones(self.N)
        with np.errstate(all="ignore"):
            self.old_obj = np.max(np.abs((F - self.Z) * self.W), axis=1)
        self.EP = None

    def step(self):
        rng, N, nr = self.rng, self.N, self.nr
        if int(np.ceil(self.FE / N)) % 10 == 0:
            with np.errstate(all="ignore"):
                new_obj = np.max(np.abs((objs(self.pop) - self.Z) * self.W), axis=1)
                delta = (self.old_obj - new_obj) / self.old_obj
            temp = delta <= 0.001
            self.Pi[~temp] = 1
            self.Pi[temp] = (0.95 + 0.05 * delta[temp] / 0.001) * self.Pi[temp]
            self.old_obj = new_obj
        off = None
        for _ in range(5):
            boundary = np.where(np.sum(self.W < 1e-3, axis=1) == 1)[0]
            I = np.concatenate([boundary, tournament(10, N // 5 - len(boundary), -self.Pi, rng=rng)]).astype(int)
            kids = []
            for i in I:
                P = self.B[i][rng.permutation(self.T)] if rng.random() < 0.9 else rng.permutation(N)
                child = self.evaluate(ga_half(self.problem, decs(self.pop[P[:2]]), rng=rng))
                kids.append(child[0])
                fo = objs(child)[0]
                self.Z = np.minimum(self.Z, fo)
                g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) * self.W[P], axis=1)
                g_new = np.max(np.abs(fo - self.Z) * self.W[P], axis=1)
                self.pop[P[np.where(g_old >= g_new)[0][:nr]]] = child[0]
            off = Population.create(kids)
        if self.FE >= self.rate_evol * self.max_FE:
            self.EP = update_ep(self.pop if self.EP is None else self.EP, off, self.nEP)
            if int(np.ceil(self.FE / N)) % (self.wag / 5) == 0:
                self.pop, self.W = _update_weight(self.pop, self.W, self.Z, self.EP, self.rate_update_weight * N)
                self.B = neighbors_of(self.W, self.T) if False else self.B
