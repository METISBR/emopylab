# emopylab 2026
"""MOEA-D-URAW (mOEA/D with uniform randomly adaptive weights).

Reference:
L. R. C. Farias and A. F. R. Araujo. Many-objective evolutionary algorithm based on decomposition
with random and adaptive weights. Proceedings of the IEEE International Conference on Systems, Mans
and Cybernetics, 2019.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga_half, neighbors_of, objs, pdist2
from algorithms.moea_d_awa.moea_d_awa import update_ep
from core.population import Population

ALGORITHM_FLAGS = {'MOEADURAW': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def uniformly_randomly_point(N, M, rng):
    """Farthest-first selection of N weight vectors from 5000 uniformly random simplex samples plus the corners."""
    W1 = np.vstack([np.eye(M), np.ones((1, M)) / M])
    W2 = rng.random((5000, M))
    W2 = W2 / W2.sum(axis=1, keepdims=True)
    while len(W1) < N:
        temp = np.sort(pdist2(W2, W1), axis=1)
        index = int(np.lexsort(temp.T[::-1])[-1])
        W1 = np.vstack([W1, W2[index]])
        W2 = np.delete(W2, index, axis=0)
    W1 = np.maximum(W1, 1e-6)
    return W1, len(W1)


def _update_weight(pop, W, Z, EP, nus):
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
        temp = 1.0 / (new_objs - Z)
    temp[np.isinf(temp)] = 0.999999
    W = np.vstack([W, temp / temp.sum(axis=1, keepdims=True)])
    return comb[sel], W


class MOEADURAW(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, delta: float = 0.9, nr: int = 2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta, self.nr = float(delta), int(nr)

    def initial_size(self):
        W, self.pop_size = uniformly_randomly_point(self.pop_size, self.M, self.rng)
        self.W = 1.0 / W / np.sum(1.0 / W, axis=1, keepdims=True)
        self.T = int(np.ceil(self.pop_size / 10))
        self.nEP = int(np.ceil(self.pop_size * 2))
        self.nus = 0.05
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.Z = objs(self.pop).min(axis=0)
        self.EP = None
        self.adapt = int(round(np.ceil(self.max_FE / self.N) * 0.05))

    def step(self):
        rng, N = self.rng, self.N
        kids = []
        for i in range(N):
            P = self.B[i][rng.permutation(self.T)] if rng.random() < self.delta else rng.permutation(N)
            child = self.evaluate(ga_half(self.problem, decs(self.pop[P[:2]]), rng=rng))
            kids.append(child[0])
            fo = objs(child)[0]
            self.Z = np.minimum(self.Z, fo)
            g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) * self.W[P], axis=1)
            g_new = np.max(np.abs(fo - self.Z) * self.W[P], axis=1)
            self.pop[P[np.where(g_old >= g_new)[0][: self.nr]]] = child[0]
        if self.FE / self.max_FE <= 0.9:
            off = Population.create(kids)
            self.EP = update_ep(self.pop if self.EP is None else self.EP, off, self.nEP)
            if self.adapt > 0 and int(np.ceil(self.FE / N)) % self.adapt == 0:
                self.pop, self.W = _update_weight(self.pop, self.W, self.Z, self.EP, self.nus * N)
                self.B = neighbors_of(self.W, self.T)
