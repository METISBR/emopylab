# emopylab 2026
"""CCGDE3 (cooperative coevolution generalized differential evolution 3).

Reference:
L. M. Antonio and C. A. Coello Coello. Use of cooperative coevolution for solving large scale
multiobjective optimization problems. Proceedings of the IEEE Congress on Evolutionary Computation,
2013, 2758-2765.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, first_front, gde3_selection, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'CCGDE3': {'constrained', 'integer', 'multi', 'real'}}


def _ccde(P1, P2, P3, lo, up, rng, CR=0.5, F=0.5, proM=1.0, disM=20.0):
    N, D = P1.shape
    off = P1.copy()
    site = rng.random((N, D)) < CR
    off[site] = off[site] + F * (P2[site] - P3[site])
    lo_b, up_b = np.broadcast_to(lo, (N, D)), np.broadcast_to(up, (N, D))
    span = up_b - lo_b
    s2 = rng.random((N, D)) < proM / D
    mu = rng.random((N, D))
    off = np.minimum(np.maximum(off, lo_b), up_b)
    t = s2 & (mu <= 0.5)
    off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lo_b[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
    t = s2 & (mu > 0.5)
    off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up_b[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return off


def _environmental_selection(pop, N):
    F, c = objs(pop), cons(pop)
    front_no, max_f = nd_sort(F, c if c.size else None, N)
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt]


class CCGDE3(LoopAlgorithm):
    """Cooperative co-evolutionary GDE3 with two randomly grouped variable subcomponents.

    NOTE (kept literally from the reference): after the second subpopulation is evolved its variables are
    re-read from the *first* subpopulation (``Dec2 = Population1.decs``)."""

    def initial_size(self):
        self.n_sub = int(np.ceil(0.8 * self.pop_size))
        return self.n_sub

    def _get_ind(self, sub, other, j):
        n, D = self.n_sub, self.D
        X = np.zeros((n, D))
        rng = self.rng
        if j == 1:
            X[:, self.index == 1] = sub
            X[:, self.index == 2] = other[rng.integers(0, len(other), size=n)]
        else:
            X[:, self.index == 2] = sub
            X[:, self.index == 1] = other[rng.integers(0, len(other), size=n)]
        return self.evaluate(X)

    def start(self):
        rng, D = self.rng, self.D
        per = D // 2
        idx = np.concatenate([np.full(per, 1), np.full(D - per, 2)])
        self.index = idx[rng.permutation(D)]
        dec = np.asarray(self.pop.get("X"), dtype=float)[: self.n_sub]
        if len(dec) < self.n_sub:
            dec = np.vstack([dec, rng.uniform(self.lower, self.upper, size=(self.n_sub - len(dec), D))])
        self.sub1, self.sub2 = dec[:, self.index == 1], dec[:, self.index == 2]
        self.pop1 = self._get_ind(self.sub1, self.sub2, 1)
        self.pop2 = self._get_ind(self.sub2, self.sub1, 2)
        self.pop = _environmental_selection(Population.merge(self.pop1, self.pop2), self.N)

    def step(self):
        rng, n = self.rng, self.n_sub
        nd1 = self.sub1[first_front(objs(self.pop1))]
        nd2 = self.sub2[first_front(objs(self.pop2))]
        lo, up = self.lower, self.upper
        for j in (1, 2):
            if j == 1:
                off_dec = _ccde(self.sub1, self.sub1[rng.integers(0, n, size=n)], self.sub1[rng.integers(0, n, size=n)],
                                lo[self.index == 1], up[self.index == 1], rng)
                off = self._get_ind(off_dec, nd2, 1)
                self.pop1 = gde3_selection(self.pop1, off, n)
                self.sub1 = np.asarray(self.pop1.get("X"), dtype=float)[:, self.index == 1]
            else:
                off_dec = _ccde(self.sub2, self.sub2[rng.integers(0, n, size=n)], self.sub2[rng.integers(0, n, size=n)],
                                lo[self.index == 2], up[self.index == 2], rng)
                off = self._get_ind(off_dec, nd1, 2)
                self.pop2 = gde3_selection(self.pop2, off, n)
                self.sub2 = np.asarray(self.pop1.get("X"), dtype=float)[:, self.index == 2]
            self.pop = _environmental_selection(Population.merge(self.pop1, self.pop2), self.N)
