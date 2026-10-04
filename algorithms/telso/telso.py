# emopylab 2026
"""TELSO (two-layer encoding learning swarm optimizer).

Reference:
S. Qi, R. Wang, T. Zhang, X. Yang, R. Sun, and L. Wang. A two-layer encoding learning swarm
optimizer based on frequent itemsets for sparse large-scale multi-objective optimization. IEEE/CAA
Journal of Automatica Sinica, 2024, 11(6): 1342-1357.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, nd_sort, objs, tournament, uniform_point, velocity
from algorithms.lmocso.lmocso import _angle, _nan_argmin, cal_fitness
from core.population import Population

ALGORITHM_FLAGS = {'TELSO': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


def _cvs(pop):
    C = cons(pop)
    return np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))


def _initial_selection(pop, Mask, N):
    C = cons(pop)
    front, maxf = nd_sort(objs(pop), C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(objs(pop), front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], Mask[nxt]


def _env_selection(pop, V, theta, Mask):
    Mask = np.vstack([Mask, Mask])
    F = objs(pop)
    N, M = F.shape
    NV = len(V)
    F = F - F.min(axis=0)
    CV = _cvs(pop)
    cosine = np.cos(_angle(V, V))
    np.fill_diagonal(cosine, 0.0)
    gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
    Angle = _angle(F, V)
    associate = _nan_argmin(Angle, 1)
    nxt = -np.ones(NV, dtype=int)
    for i in np.unique(associate):
        c1 = np.where((associate == i) & (CV == 0))[0]
        c2 = np.where((associate == i) & (CV != 0))[0]
        if len(c1):
            with np.errstate(all="ignore"):
                apd = (1 + M * theta * Angle[c1, i] / gamma[i]) * np.sqrt(np.sum(F[c1] ** 2, axis=1))
            nxt[i] = c1[_nan_argmin(apd, 0)]
        elif len(c2):
            nxt[i] = c2[int(np.argmin(CV[c2]))]
    a = nxt[nxt >= 0]
    return pop[a], Mask[a]


class TELSO(LoopAlgorithm):
    """Two-stage evolutionary large-scale sparse optimizer: variable scores from an initial probe seed the sparse
    masks; a swarm then moves the decision vectors while masks are aligned between particles with a growing
    share of their common bits."""

    def _initialize_infill(self):
        self.V, self.pop_size = uniform_point(self.pop_size, self.M)
        N, D, rng = self.N, self.D, self.rng
        lo, up = self.lower, self.upper
        real = bool(np.all(self.encoding == 1))
        TDec, TMask, TPop = [], [], []
        DF = np.zeros(D)
        for _ in range(1 + 4 * int(real)):
            Dec = lo + rng.random((D, D)) * (up - lo) if real else np.ones((D, D))
            Mask = np.eye(D)
            P = self.evaluate(Dec * Mask)
            TDec.append(Dec), TMask.append(Mask), TPop.append(P)
            C = cons(P)
            DF = DF + nd_sort(np.hstack([objs(P), C]) if C.size else objs(P), None, np.inf)[0]
        Dec = lo + rng.random((N, D)) * (up - lo) if real else np.ones((N, D))
        Mask = np.zeros((N, D))
        for i in range(N):
            Mask[i, tournament(2, int(np.ceil(rng.random() * D)), DF, rng=rng)] = 1
        P = self.evaluate(Dec * Mask)
        pop, Mask = _initial_selection(Population.merge(P, *TPop), np.vstack([Mask] + TMask), N)
        self.Mask = Mask
        pop, self.Mask = _env_selection(pop, self.V, (self.FE / self.max_FE) ** 2, self.Mask)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _operator(self, pop, Mask):
        rng, FE, mx = self.rng, self.FE, self.max_FE
        X = decs(pop)
        N, D = X.shape
        Vel = velocity(pop)
        Mask = Mask.copy()
        offdec, offvel = [], []
        for i in range(N - 2):
            ri = rng.integers(i + 1, N, size=2)
            r1, r2, r3 = rng.random(), rng.random(), rng.random()
            # the swarm's own-velocity term uses the (i)-th element of the velocity matrix (column-major order)
            v = r1 * Vel.reshape(-1, order="F")[i] + r2 * (X[ri[0]] - X[i]) + r3 * (X[ri[1]] - X[i])
            offdec.append(X[i] + v)
            offvel.append(v)
            p1, p2 = Mask[ri[0]], Mask[ri[1]]
            same = ((p1 == p2) & (p2 == 1)) if rng.random() < 0.5 else ((p1 == p2) & (p2 == 0))
            cols = np.where(same)[0]
            keep = int(np.floor(len(cols) * FE / mx))
            cols = cols[:keep]
            if len(cols):
                Mask[i, cols] = Mask[ri[0], cols]
        offdec += [X[N - 2], X[N - 1]] if N >= 2 else []
        offvel += [Vel[N - 2], Vel[N - 1]] if N >= 2 else []
        off = self.evaluate(np.array(offdec) * Mask[: len(offdec)], V=np.array(offvel))
        return off, Mask

    def step(self):
        fit = cal_fitness(objs(self.pop))
        index = np.argsort(fit, kind="stable")
        pop, Mask = self.pop[index], self.Mask[index]
        off, Mask = self._operator(pop, Mask)
        self.pop, self.Mask = _env_selection(Population.merge(pop, off), self.V, (self.FE / self.max_FE) ** 2, Mask)
