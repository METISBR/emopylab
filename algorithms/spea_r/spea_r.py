# emopylab 2026
"""SPEA-R (strength Pareto evolutionary algorithm based on reference direction).

Reference:
S. Jiang and S. Yang. A strength Pareto evolutionary algorithm based on reference direction for
multiobjective and many-objective optimization. IEEE Transactions on Evolutionary Computation, 2017,
21(3): 329-346.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, angle_matrix, decs, dominance_matrix, ga_half, first_front, objs, pdist2, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'SPEAR': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _objective_normalization(F):
    nd = first_front(F)
    zmin, zmax = F[nd].min(axis=0), F[nd].max(axis=0)
    with np.errstate(all="ignore"):
        return (F - zmin) / (zmax - zmin)


def _fitness_assignment(Ei, F, angle, theta):
    N = len(F)
    dom = dominance_matrix(F)
    Sl, Rl = np.zeros(N), np.zeros(N)
    for i in np.unique(Ei):
        local = np.where(Ei == i)[0]
        Sl[local] = dom[np.ix_(local, local)].sum(axis=1)
        # the reference implementation indexes the dominance column with the niche id itself
        col = dom[local, i] if i < N else np.zeros(len(local), bool)
        Rl[local] = Sl[local[col]].sum()
    Sg = dom.sum(axis=1)
    Rg = np.array([Sg[dom[:, i]].sum() for i in range(N)], dtype=float)
    D = angle / (angle + theta)
    FV = np.zeros(N)
    for i in np.unique(Ei):
        local = np.where(Ei == i)[0]
        FV[local] = Rl[local] + D[local] + (0 if len(local) == 1 else Rg[local])
    return FV


class SPEAR(LoopAlgorithm):
    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        cosine = np.cos(angle_matrix(self.W))
        np.fill_diagonal(cosine, 0.0)
        self.theta = float(np.max(np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)))
        return self.pop_size

    def _mating_selection(self, F, K):
        N, rng = len(F), self.rng
        dis = pdist2(F, F)
        np.fill_diagonal(dis, np.inf)
        pool = np.zeros(N, dtype=int)
        for i in range(N):
            cand = rng.permutation(N)[: min(K, N)]
            pool[i] = cand[int(np.argmin(dis[i, cand]))]
        return pool

    def _environmental_selection(self, pop, Ei, FV):
        N = self.N
        Ei = Ei.copy()
        choose = []
        while len(choose) < N:
            H = []
            for i in np.unique(Ei):
                if i >= 0:
                    local = np.where(Ei == i)[0]
                    H.append(int(local[np.argmin(FV[local])]))
            H = np.asarray(H, dtype=int)
            if len(H) == 0:
                break
            if self.FE >= self.max_FE and np.any(FV[H] < 1):
                H = H[FV[H] < 1]
            Ei[H] = -1
            if len(choose) + len(H) <= N:
                choose += list(H)
            else:
                choose += list(H[np.argsort(FV[H], kind="stable")[: N - len(choose)]])
        return pop[np.asarray(choose, dtype=int)]

    def step(self):
        N = self.N
        pool = self._mating_selection(objs(self.pop), 20)
        parents = np.vstack([decs(self.pop), decs(self.pop[pool])])
        off = self.evaluate(ga_half(self.problem, parents, rng=self.rng))
        merged = Population.merge(self.pop, off)
        Q = _objective_normalization(objs(merged))
        ang = angle_matrix(Q, self.W)                    # (N2, NW)
        Ei = np.argmin(ang, axis=1)
        angle = ang[np.arange(len(Q)), Ei]
        FV = _fitness_assignment(Ei, Q, angle, self.theta)
        self.pop = self._environmental_selection(merged, Ei, FV)
