# emopylab 2026
"""MO_Ring_PSO_SCD (multiobjective PSO using ring topology and special crowding distance).

Reference:
C. Yue, B. Qu, and J. Liang. A multiobjective particle swarm optimizer using ring topology for
solving multimodal multiobjective problems. IEEE Transactions on Evolutionary Computation, 2018,
22(5): 805-817.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, objs, velocity
from core.population import Population

ALGORITHM_FLAGS = {'MO_Ring_PSO_SCD': {'integer', 'multi', 'multimodal', 'real'}}


def _modified_crowding(P, front):
    N, M = P.shape
    cd = np.zeros(N)
    for f in np.unique(front[np.isfinite(front)]):
        idx = np.where(front == f)[0]
        fmax, fmin = P[idx].max(axis=0), P[idx].min(axis=0)
        for i in range(M):
            rank = idx[np.argsort(P[idx, i], kind="stable")]
            cd[rank[0]] += 1
            for j in range(1, len(idx) - 1):
                if fmax[i] == fmin[i]:
                    cd[rank[j]] += 1
                else:
                    cd[rank[j]] += (P[rank[j + 1], i] - P[rank[j - 1], i]) / (fmax[i] - fmin[i])
    return cd


def scd_sort(pop, N):
    """Non-dominated sorting with the special crowding distance (objective- and decision-space, whichever is larger
    unless both are below the front average).  Returns the survivors (original order), their fronts and distances."""
    N = min(N, len(pop))
    front, maxf = nd_sort(objs(pop), None, N)
    nxt = front < maxf
    so, sd = _modified_crowding(objs(pop), front), _modified_crowding(decs(pop), front)
    scd = np.maximum(so, sd)
    for i in range(1, int(maxf) + 1):
        fr = np.where(front == i)[0]
        rep = (so[fr] <= so[fr].mean()) & (sd[fr] <= sd[fr].mean())
        scd[fr[rep]] = np.minimum(so[fr[rep]], sd[fr[rep]])
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-scd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], scd[nxt]


def _sorted_archive(pop, N):
    p, front, scd = scd_sort(pop, N)
    return p[np.lexsort((-scd, front))]


class MO_Ring_PSO_SCD(LoopAlgorithm):
    """Multimodal multi-objective ring-topology PSO: every particle keeps a personal-best archive and a ring
    neighbourhood-best archive, both ranked by non-dominated sorting with the special crowding distance."""

    def __init__(self, pop_size: int = 100, n_PBA: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.n_pba = int(n_PBA)

    def _initialize_infill(self):
        N, D, rng = self.N, self.D, self.rng
        lo, up = self.lower, self.upper
        mv = 0.5 * (up - lo)
        dec = lo + (up - lo) * rng.random((N, D))
        vel = -mv + 2 * mv * rng.random((N, D))
        return self.evaluate(dec, V=vel)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.PBA = [infills[[i]] for i in range(len(infills))]
        self.NBA = [infills[[i]] for i in range(len(infills))]
        self._set_optimum()

    def _update_nba(self):
        n = len(self.NBA)
        for i in range(n):
            neigh = Population.merge(self.PBA[(i - 1) % n], self.PBA[i], self.PBA[(i + 1) % n], self.NBA[i])
            self.NBA[i] = _sorted_archive(neigh, 3 * self.n_pba)

    def _operator(self):
        pop, rng = self.pop, self.rng
        X, V = decs(pop), velocity(pop)
        N, D = X.shape
        Pb = np.array([np.asarray(self.PBA[i][0].X, float) for i in range(N)])
        Nb = np.array([np.asarray(self.NBA[i][0].X, float) for i in range(N)])
        vel = 0.7298 * V + 2.05 * rng.random((N, D)) * (Pb - X) + 2.05 * rng.random((N, D)) * (Nb - X)
        delta = (self.upper - self.lower) / 2
        vel = np.maximum(np.minimum(vel, delta), -delta)
        dec = X + vel
        lo, up = np.broadcast_to(self.lower, dec.shape), np.broadcast_to(self.upper, dec.shape)
        t = dec < lo
        dec[t] = lo[t] + 0.25 * (up[t] - lo[t]) * rng.random(int(t.sum()))
        t = dec > up
        dec[t] = up[t] - 0.25 * (up[t] - lo[t]) * rng.random(int(t.sum()))
        return self.evaluate(dec, V=vel)

    def step(self):
        self._update_nba()
        self.pop = self._operator()
        for i in range(len(self.PBA)):
            self.PBA[i] = _sorted_archive(Population.merge(self.PBA[i], self.pop[[i]]), self.n_pba)
        if self.FE >= self.max_FE:
            allnba = Population.merge(*self.NBA)
            p, front, _ = scd_sort(allnba, self.N)
            self.pop = p[front == 1]
