# emopylab 2026
"""one-by-one EA (many-objective evolutionary algorithm using a one-by-one selection).

Reference:
Y. Liu, D. Gong, J. Sun, and Y. Jin. A many-objective evolutionary algorithm using a one-by-one
selection strategy. IEEE Transactions on Cybernetics, 2017, 47(9): 2689-2702.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, ga, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'onebyoneEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _environmental_selection(pop, zeta, zmin, K):
    F = objs(pop)
    N, M = F.shape
    Q, Qs, Qth, Qd = list(range(N)), [], [], []
    rank = np.full(N, np.inf)
    nrank = 1
    zmin = np.minimum(zmin, F.min(axis=0))
    G = F - zmin
    c = np.sqrt(np.sum(G ** 2, axis=1))
    d = cosine_distance(G)

    def select(fitness):
        nonlocal Q, Qth, Qd
        x = int(np.argmin(fitness[Q]))
        xu = Q.pop(x)
        Qs.append(xu)
        # de-emphasise near-duplicates in angle, then dominated solutions
        near = [q for q in Q if d[q, xu] < zeta]
        Qth += near
        Q = [q for q in Q if d[q, xu] >= zeta]
        dom = [q for q in Q if np.all(G[q] >= G[xu])]
        Qd += dom
        Q = [q for q in Q if not np.all(G[q] >= G[xu])]
        rank[xu] = nrank

    for m in range(M):
        if Q:
            select(np.sqrt(np.sum(np.delete(G, m, axis=1) ** 2, axis=1)))
    while Q:
        select(c)
    r = len(Qs) / K
    Nd = len(Qd)
    while len(Qs) < K:
        if not Q:
            Q, Qth, Qd = Qth + Qd, [], []
            nrank += 1
        select(c)
    sel = np.asarray(Qs[:K])
    if Nd <= K:
        zeta = zeta * np.exp((r - 1) / M)
    return pop[sel], rank[sel], zeta, zmin


def _mating_selection(F, rank, rng):
    d = np.sort(cosine_distance(F), axis=1)
    dk = 1.0 / (d[:, 1: int(np.ceil(len(F) / 10))].sum(axis=1) + 1.0)
    return tournament(2, len(F), rank, dk, rng=rng)


class onebyoneEA(LoopAlgorithm):
    def start(self):
        self.zmin = objs(self.pop).min(axis=0)
        self.rank = np.ones(len(self.pop))
        self.zeta = 1.0

    def step(self):
        pool = _mating_selection(objs(self.pop), self.rank, self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.rank, self.zeta, self.zmin = _environmental_selection(
            Population.merge(self.pop, off), self.zeta, self.zmin, self.N)
