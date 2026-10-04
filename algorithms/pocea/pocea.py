# emopylab 2026
"""POCEA (paired offspring generation based constrained evolutionary algorithm).

Reference:
C. He, R. Cheng, Y. Tian, X. Zhang, K. C. Tan, and Y. Jin. Paired offspring generation for
constrained large-scale multiobjective optimization. IEEE Transactions on Evolutionary Computation,
2021, 25(3): 448-462.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, uniform_point
from algorithms.community_utils.base import objs
from algorithms.lmocso.lmocso import LMOCSO, _angle, _nan_argmin
from core.population import Population

ALGORITHM_FLAGS = {'POCEA': {'constrained', 'integer', 'large', 'multi', 'real'}}


def _cv(pop):
    C = cons(pop)
    return np.sum(np.maximum(C, 0), axis=1) if C.size else np.zeros(len(pop))


def rvea_selection(pop, V, popsize, theta, rng):
    F = objs(pop)
    N = len(F)
    NV = len(V)
    F = F - F.min(axis=0)
    CV = _cv(pop)
    cosine = np.cos(_angle(V, V))
    np.fill_diagonal(cosine, 0.0)
    gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
    Angle = _angle(F, V)
    associate = _nan_argmin(Angle, 1)
    nxt = -np.ones(NV, dtype=int)
    pf = np.sum(CV[rng.integers(0, N, N)] < 1e-6) / N
    uniq = np.unique(associate)
    for i in uniq:
        cv = CV[associate == i]
        Ns = len(cv)
        subN = int(np.ceil(popsize / len(uniq)))
        eps = cv.max() if Ns < subN else cv.min() * (1 - pf) + cv.mean() * pf
        c1 = np.where((associate == i) & (CV <= eps))[0]
        c2 = np.where((associate == i) & (CV > eps))[0]
        if len(c1):
            with np.errstate(all="ignore"):
                apd = (1 + theta * Angle[c1, i] / gamma[i]) * np.sqrt(np.sum(F[c1] ** 2, axis=1))
            nxt[i] = c1[_nan_argmin(apd, 0)]
        elif len(c2):
            nxt[i] = c2[int(np.argmin(CV[c2]))]
    return pop[nxt[nxt >= 0]]


class POCEA(LoopAlgorithm):
    """Constrained competitive-swarm EA with sub-populations built around sub-reference vectors, epsilon-based
    constraint-handling pairwise competition and RVEA-style environmental selection."""

    _operator = LMOCSO._operator

    def __init__(self, pop_size: int = 100, k: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.k = int(k)

    def start(self):
        self.V0, self.pop_size = uniform_point(self.pop_size, self.M)
        self.Vs0, self.L = uniform_point(int(np.floor(self.pop_size / self.k)), self.M)
        self.V, self.Vs = self.V0, self.Vs0

    def _association(self, pop):
        F = objs(pop)
        F = F - F.min(axis=0)
        dis = np.sum(F ** 2, axis=1)
        theta = _angle(F, self.Vs)
        index = np.argsort(np.where(np.isnan(theta), np.inf, theta), axis=0, kind="stable")
        return index[: min(self.k, len(pop))], theta, dis

    def _chp(self, p1, p2, eps):
        c1, c2 = _cv(p1)[0], _cv(p2)[0]
        if max(c1, c2) <= eps:
            first = np.linalg.norm(objs(p1)[0]) < np.linalg.norm(objs(p2)[0])
        else:
            first = c1 < c2
        return (p1, p2) if first else (p2, p1)

    def step(self):
        rng, k, L, pop, N = self.rng, self.k, self.L, self.pop, self.pop_size
        INDEX, THETA, DIS = self._association(pop)
        CV = _cv(pop)
        rf = np.sum(CV < 1e-6) / len(pop)
        winners, losers = [], []
        for i in range(L):
            idx = INDEX[:, i]
            sub = pop[idx]
            theta = THETA.reshape(-1, order="F")[idx]      # linear indexing: first column of the angle matrix
            if np.mean(theta) >= np.pi / L / 2:
                sel = np.argsort(DIS, kind="stable")[: min(k, len(DIS))]
                sub = Population.merge(sub, pop[sel])
                eps = CV[np.concatenate([idx, sel])].max()
            else:
                eps = CV[idx].min() * (1 - rf) + CV[idx].mean() * rf
            n = len(sub)
            r1, r2 = (0, 0) if n < 2 else (int(rng.permutation(n)[0]), int(rng.permutation(n)[0]))
            w, l = self._chp(sub[[r1]], sub[[r2]], eps)
            winners.append(w[0]), losers.append(l[0])
        off = self._operator(Population.create(losers), Population.create(winners))
        self.pop = rvea_selection(Population.merge(pop, off), self.V, N, (self.FE / self.max_FE) ** 2, rng)
        if int(np.ceil(self.FE / N)) % int(np.ceil(0.1 * self.max_FE / N)) == 0:
            rng_obj = objs(self.pop).max(axis=0) - objs(self.pop).min(axis=0)
            self.V, self.Vs = self.V0 * rng_obj, self.Vs0 * rng_obj
