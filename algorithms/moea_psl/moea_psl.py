# emopylab 2026
"""MOEA-PSL (multi-objective evolutionary algorithm based on Pareto optimal subspace).

Reference:
Y. Tian, C. Lu, X. Zhang, K. C. Tan, and Y. Jin. Solving large-scale multi-objective optimization
problems with sparse optimal solutions via unsupervised neural networks. IEEE Transactions on
Cybernetics, 2021, 51(6): 3115-3128.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, nd_sort, objs, tournament, uniform_point
from algorithms.community_utils.nn import DAE, RBM
from core.population import Population

ALGORITHM_FLAGS = {'MOEAPSL': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


def _binary_crossover(P1, P2, rng):
    k = rng.random(P1.shape) < 0.5
    o1, o2 = P1.copy(), P2.copy()
    o1[k], o2[k] = P2[k], P1[k]
    return np.vstack([o1, o2])


def _binary_mutation(off, rng):
    site = rng.random(off.shape) < 1.0 / off.shape[1]
    off = off.copy()
    off[site] = ~off[site]
    return off


def _real_crossover(P1, P2, rng, disC=20):
    N, D = P1.shape
    mu = rng.random((N, D))
    beta = np.zeros((N, D))
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
    beta = beta * (-1.0) ** rng.integers(0, 2, (N, D))
    return np.vstack([(P1 + P2) / 2 + beta * (P1 - P2) / 2, (P1 + P2) / 2 - beta * (P1 - P2) / 2])


def _real_mutation(off, lo, up, rng, disM=20):
    n, D = off.shape
    Lo, Up = np.tile(lo, (n, 1)), np.tile(up, (n, 1))
    site = rng.random((n, D)) < 1.0 / D
    mu = rng.random((n, D))
    off = np.minimum(np.maximum(off, Lo), Up)
    span = Up - Lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - Lo[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (Up[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return np.minimum(np.maximum(off, Lo), Up)


class MOEAPSL(LoopAlgorithm):
    """Sparse EA with Pareto-optimal-subspace learning: a restricted Boltzmann machine (masks) and a denoising
    autoencoder (decisions) provide a learned low-dimensional space where crossover is performed for a share
    ``rho`` of the offspring; ``rho`` follows the success ratio of the learned offspring."""

    def _initialize_infill(self):
        N, D, rng = self.N, self.D, self.rng
        lo, up, enc = self.lower, self.upper, self.encoding
        P, _ = uniform_point(N, D, "Latin")
        Dec = P * (up - lo) + lo
        Dec[:, enc == 4] = 1
        Mask = uniform_point(N, D, "Latin")[0] > 0.5
        pop = self.evaluate(Dec * Mask)
        pop, self.Dec, self.Mask, self.front, self.crowd, _ = self._select(pop, Dec, Mask, N, 0, 0)
        self.rho = 0.5
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _select(self, pop, Dec, Mask, N, length, num):
        success = np.zeros(len(pop), bool)
        F = objs(pop)
        _, uni = np.unique(F, axis=0, return_index=True)
        if len(uni) == 1:
            _, uni = np.unique(decs(pop), axis=0, return_index=True)
        pop, Dec, Mask = pop[uni], Dec[uni], Mask[uni]
        N = min(N, len(pop))
        C = cons(pop)
        front, maxf = nd_sort(objs(pop), C if C.size else None, N)
        nxt = front < maxf
        cd = crowding(objs(pop), front)
        last = np.where(front == maxf)[0]
        nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
        success[uni[nxt]] = True
        s1 = success[length:length + num].sum()
        s2 = success[length + num:].sum()
        ratio = float(np.clip((s1 + 1e-6) / (s1 + s2 + 1e-6), 0.1, 0.9))
        return pop[nxt], Dec[nxt], Mask[nxt], front[nxt], cd[nxt], ratio

    def _train(self, Mask, Dec, real):
        rng = self.rng
        allzero, allone = ~Mask.any(axis=0), Mask.all(axis=0)
        other = ~allzero & ~allone
        K = int(np.sum(np.mean(np.abs(Mask[:, other] * Dec[:, other]) > 1e-6, axis=0) > rng.random(int(other.sum()))))
        K = min(max(K, 1), len(Mask))
        rbm = RBM(int(other.sum()), K, 10, 1, 0, 0.5, 0.1, rng)
        rbm.train(Mask[:, other])
        dae = None
        if real:
            dae = DAE(Dec.shape[1], K, 10, len(Dec), 0.5, 0.5, 0.1, rng)
            dae.train(Dec)
        return rbm, dae, allzero, allone

    def _operator(self, ParentDec, ParentMask, rbm, dae, site, allzero, allone):
        rng, D = self.rng, self.D
        h = len(ParentMask) // 2
        P1m, P2m, P1d, P2d = ParentMask[:h], ParentMask[h:], ParentDec[:h], ParentDec[h:]
        if site.any():
            other = ~allzero & ~allone
            t = _binary_crossover(rbm.reduce(P1m[site][:, other]), rbm.reduce(P2m[site][:, other]), rng)
            t = rbm.recover(t)
            OffMask = np.zeros((len(t), D), bool)
            OffMask[:, other] = t
            OffMask[:, allone] = True
        else:
            OffMask = np.zeros((0, D), bool)
        OffMask = np.vstack([OffMask, _binary_crossover(P1m[~site], P2m[~site], rng)])
        OffMask = _binary_mutation(OffMask, rng)
        enc = self.encoding
        if np.any(enc != 4):
            if site.any():
                d = dae.recover(_real_crossover(dae.reduce(P1d[site]), dae.reduce(P2d[site]), rng))
            else:
                d = np.zeros((0, D))
            OffDec = np.vstack([d, _real_crossover(P1d[~site], P2d[~site], rng)])
            OffDec = _real_mutation(OffDec, self.lower, self.upper, rng)
            OffDec[:, enc == 4] = 1
        else:
            OffDec = np.ones(OffMask.shape)
        return OffDec, OffMask

    def step(self):
        N, rng = self.N, self.rng
        site = self.rho > rng.random(int(np.ceil(N / 2)))
        real = bool(np.any(self.encoding != 4))
        if site.any():
            rbm, dae, allzero, allone = self._train(self.Mask, self.Dec, real)
        else:
            rbm = dae = allzero = allone = None
        pool = tournament(2, int(np.ceil(N / 2)) * 2, self.front, -self.crowd, rng=rng)
        OffDec, OffMask = self._operator(self.Dec[pool], self.Mask[pool], rbm, dae, site, allzero, allone)
        off = self.evaluate(OffDec * OffMask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd, ratio = self._select(
            Population.merge(self.pop, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), N, len(self.pop), 2 * int(site.sum()))
        self.rho = (self.rho + ratio) / 2
