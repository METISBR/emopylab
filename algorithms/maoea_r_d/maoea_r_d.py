# emopylab 2026
"""MaOEA-R&D (many-objective evolutionary algorithm based on objective space reduction).

Reference:
Z. He and G. G. Yen. Many-objective evolutionary algorithm: Objective space reduction and diversity
improvement. IEEE Transactions on Evolutionary Computation, 2016, 20(1): 145-160.
"""

from __future__ import annotations

import itertools

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'MaOEARD': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def classification(pop, W):
    """Split the population into M equally sized classes, one per extreme direction (ASF-based assignment with
    re-assignment of the excess of over-full classes)."""
    F = objs(pop)
    N, M = F.shape
    Z = F.min(axis=0)
    ASF = np.stack([np.max((F - Z) / W[i], axis=1) for i in range(M)], axis=1)
    cls = np.zeros(N, dtype=int)
    external = np.ones(N, bool)
    lacking = np.ones(M, bool)
    share = N / M
    for _ in range(10000):
        if not lacking.any():
            break
        lacks = np.where(lacking)[0]
        sub = np.argmin(ASF[np.ix_(np.where(external)[0], lacks)], axis=1)
        cls[external] = lacks[sub] + 1
        external = np.zeros(N, bool)
        for i in range(M):
            idx = np.where(cls == i + 1)[0]
            if len(idx) > share:
                order = np.argsort(ASF[idx, i], kind="stable")
                external[idx[order[int(share):]]] = True
            lacking[i] = len(idx) < share
    return [pop[cls == i + 1] for i in range(M)], Z


def update_tp(pop, W):
    subs, Z = classification(pop, W)
    ext = []
    for i, s in enumerate(subs):
        asf = np.max((objs(s) - Z) / W[i], axis=1)
        ext.append(s[[int(np.argmin(asf))]])
    return Population.merge(*ext)


def _diversity_operator(P, N, fmax, fmin):
    n1 = len(P)
    nxt = np.arange(n1)
    with np.errstate(all="ignore"):
        gloc = N * (P - fmin) / (fmax - fmin)
    A = np.sqrt(np.maximum(((gloc[:, None, :] - gloc[None, :, :]) ** 2).sum(axis=2), 0))
    np.fill_diagonal(A, np.inf)
    while len(nxt) > N:
        sub = A[np.ix_(nxt, nxt)]
        near = np.argmin(sub, axis=1)
        dis = sub[np.arange(len(nxt)), near]
        si = [[dis[k], k, near[k]] for k in range(len(nxt))]
        elim = np.zeros(len(nxt), bool)
        while si and len(nxt) - elim.sum() > N:
            i1 = int(np.argmin([r[0] for r in si]))
            i1 = si[i1][1]
            elim[i1] = True
            si = [r for r in si if r[1] != i1 and r[2] != i1]
        nxt = nxt[~elim]
    return nxt


def environmental_selection(pop, N, TP):
    F = objs(pop)
    tpmax = TP.max(axis=0)
    sin = np.where(np.all(F <= tpmax, axis=1))[0]
    if len(sin) > N:
        snd = sin[nd_sort(F[sin], None, 1)[0] == 1]
        if len(snd) > N:
            nxt = snd[_diversity_operator(F[snd], N, np.vstack([F, TP]).max(axis=0), np.vstack([F, TP]).min(axis=0))]
        else:
            sd = np.setdiff1d(sin, snd)
            md = np.array([np.min(np.linalg.norm(F[snd] - F[i], axis=1)) for i in sd])
            nxt = np.concatenate([snd, sd[np.argsort(-md, kind="stable")[: N - len(snd)]]])
    else:
        pairs = list(itertools.combinations(range(len(TP)), 2))
        mid = np.array([(TP[a] + TP[b]) / 2 for a, b in pairs])
        pts = np.vstack([mid, TP])
        sout = np.setdiff1d(np.arange(len(F)), sin)
        md = np.array([np.min(np.linalg.norm(pts - F[i], axis=1)) for i in sout])
        nxt = np.concatenate([sin, sout[np.argsort(md, kind="stable")[: N - len(sin)]]]).astype(int)
    return pop[nxt.astype(int)]


class MaOEARD(LoopAlgorithm):
    """Many-objective EA with reduction and decomposition into M single-direction subpopulations: the first half of
    the budget evolves the M extreme-direction subpopulations, the second half evolves one population driven by
    the extreme (target) points, keeping solutions inside their bounding box and diversifying by grid crowding."""

    def initial_size(self):
        M = self.M
        self.pop_size = max(int(np.ceil(self.pop_size / M)) * M, 2 * M)
        return self.pop_size

    def step(self):
        N, M, rng, pr = self.N, self.M, self.rng, self.problem
        W = np.zeros((M, M)) + 1e-6
        np.fill_diagonal(W, 1.0)
        pop = self.pop
        k = N // M
        while self.not_terminated(pop) and self.FE < self.max_FE / 2:
            subs, Z = classification(pop, W)
            for i in range(M):
                pool = rng.integers(0, k, k)
                off = self.evaluate(ga(pr, decs(subs[i][pool]), rng=rng))
                s = Population.merge(subs[i], off)
                asf = np.max((objs(s) - Z) / W[i], axis=1)
                subs[i] = s[np.argsort(asf, kind="stable")[:k]]
            pop = Population.merge(*subs)
        self.not_terminated(pop)
        TP = update_tp(pop, W)
        X = np.tile(decs(TP), (k - 1, 1))
        X = X * (1 + rng.standard_normal(X.shape) / 5)
        pop = Population.merge(TP, self.evaluate(X))
        while self.not_terminated(pop):
            pool = rng.integers(0, N, N)
            off = self.evaluate(ga(pr, decs(pop[pool]), rng=rng))
            pop = environmental_selection(Population.merge(pop, off), N, objs(TP))
            TP = update_tp(Population.merge(pop, TP), W)
        self.pop = pop
