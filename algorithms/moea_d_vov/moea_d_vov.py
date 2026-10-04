# emopylab 2026
"""MOEA-D-VOV (mOEA/D with virtual objective vectors).

Reference:
T. Takagi, K. Takadama, and H. Sato. Weight vector arrangement using virtual objective vectors in
decomposition-based MOEA. Proceedings of the IEEE Congress on Evolutionary Computation, 2021,
1462-1469.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, first_front, ga_half, neighbors_of, objs, uniform_point

ALGORITHM_FLAGS = {'MOEADVOV': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _range_normalize(F):
    span = F.max(axis=0) - F.min(axis=0)
    return (F - F.min(axis=0)) / np.where(span > 0, span, 1.0)


def _generate_vov(F, theta):
    F = _range_normalize(F)
    W, _ = uniform_point(20000, F.shape[1], "ILD")
    norm_w, norm_f = np.linalg.norm(W, axis=1), np.linalg.norm(F, axis=1)
    with np.errstate(all="ignore"):
        cos = (W @ F.T) / norm_w[:, None] / norm_f[None, :]                       # (N2, N)
        cos = np.where(np.isfinite(cos), cos, 0.0)
        d2 = norm_f[None, :] * np.sqrt(np.maximum(0.0, 1 - cos ** 2))
    I = np.argmin(d2, axis=1)
    mind2 = d2[np.arange(len(W)), I]
    d1 = norm_f[I] * cos[np.arange(len(W)), I]
    vov = W * (d1 / norm_w)[:, None]
    return vov[mind2 < theta]


def _update_weight(F, vov, N):
    fmin, fmax = F.min(axis=0), F.max(axis=0)
    obj = np.unique(np.vstack([_range_normalize(F), vov]), axis=0)
    lp = (np.abs(obj[:, None, :] - obj[None, :, :]) ** 0.5).sum(axis=2) ** 2
    choose = np.zeros(len(obj), bool)
    choose[np.argmin(cosine_distance(obj, np.eye(obj.shape[1])), axis=0)] = True
    while choose.sum() < N:
        remain = np.where(~choose)[0]
        rho = int(np.argmax(lp[np.ix_(remain, np.where(choose)[0])].min(axis=1)))
        choose[remain[rho]] = True
    W = obj[choose] * (fmax - fmin)
    B = neighbors_of(W, int(np.ceil(N / 10)))
    return W / np.abs(W).sum(axis=1, keepdims=True), B


class MOEADVOV(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, G: int = 100, C: int = 9, theta: float = 0.02, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.G, self.C, self.theta = int(G), int(C), float(theta)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M, "ILD")
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.Z = objs(self.pop).min(axis=0)
        self.BP = self.N * self.G * self.C
        self.EP = self.pop

    def step(self):
        rng, N, T = self.rng, self.N, self.T
        kids = []
        for i in range(N):
            P = self.B[i][rng.permutation(self.B.shape[1])]
            child = self.evaluate(ga_half(self.problem, decs(self.pop[P[:2]]), rng=rng))
            kids.append(child[0])
            fc = objs(child)[0]
            self.Z = np.minimum(self.Z, fc)
            g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) / self.W[P], axis=1)
            g_new = np.max(np.abs(fc - self.Z) / self.W[P], axis=1)
            self.pop[P[g_old >= g_new]] = child[0]
        if self.FE <= self.BP:
            from core.population import Population
            self.EP = Population.merge(self.EP, Population.create(kids))
            self.EP = self.EP[first_front(objs(self.EP))]
            if len(self.EP) > 5000:
                self.EP = self.EP[len(self.EP) - 5000:]
            if int(np.ceil(self.FE / N)) % self.G == 0:
                Fe = objs(self.EP)
                vov = _generate_vov(Fe, self.theta)
                self.W, self.B = _update_weight(Fe, vov, N)
                obj = np.abs(Fe - self.Z)
                for i in range(N):
                    self.pop[i] = self.EP[int(np.argmin(np.max(obj / self.W[i], axis=1)))]
