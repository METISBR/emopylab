# emopylab 2026
"""MOEA-DD (many-objective evolutionary algorithm based on dominance and).

Reference:
K. Li, K. Deb, Q. Zhang, and S. Kwong. An evolutionary many-objective optimization algorithm based
on dominance and decomposition. IEEE Transactions Evolutionary Computation, 2015, 19(5): 694-716.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, cosine_distance, decs, ga_half, nd_sort, objs, pdist2, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MOEADD': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _weak_dom(P):
    """WD[j, i] is True when solution j is no worse than solution i in every objective."""
    return np.all(P[:, None, :] <= P[None, :, :], axis=2)


def update_front(P, front, x=None):
    """Incremental non-dominated front update: insertion of the last row of ``P`` (``x`` omitted) or removal of
    solution ``x`` (0-based).  Ranks follow the weak-dominance test of the incremental scheme."""
    n = len(P)
    WD = _weak_dom(P)
    if x is None:
        fn = np.concatenate([np.asarray(front, dtype=float), [0.0]])
        move = np.zeros(n, bool)
        move[n - 1] = True
        cur = 1
        while True:
            members = np.where(fn[: n - 1] == cur)[0]
            if len(members) and np.any(WD[members, n - 1]):
                cur += 1
            else:
                break
        while move.any():
            cand = np.where(fn == cur)[0]
            nxt = np.zeros(n, bool)
            if len(cand):
                nxt[cand] = np.any(WD[np.where(move)[0]][:, cand], axis=0)
            fn[move] = cur
            cur += 1
            move = nxt
        return fn
    fn = np.asarray(front, dtype=float).copy()                # one entry longer than P: solution x is already gone from P
    move = np.zeros(len(fn), bool)
    move[x] = True
    cur = fn[x] + 1
    while move.any():
        mv = np.where(move[:n])[0]
        nxt = np.zeros(len(fn), bool)
        cand = np.where(fn[:n] == cur)[0]
        if len(cand) and len(mv):
            nxt[cand] = np.any(WD[mv][:, cand], axis=0)
        prev = np.where((fn[:n] == cur - 1) & ~move[:n])[0]
        hit = np.where(nxt)[0]
        if len(hit) and len(prev):
            nxt[hit] = ~np.any(WD[prev][:, hit], axis=0)
        fn[move] = cur - 2
        cur += 1
        move = nxt
    return np.delete(fn, x)


def _pbi(F, W, region, Z, sub):
    out = np.zeros(len(F))
    if sub.any():
        w = W[region[sub]]
        nw = np.linalg.norm(w, axis=1)
        d1 = np.abs(np.sum((F[sub] - Z) * w, axis=1)) / nw
        d2 = np.linalg.norm(F[sub] - (Z + w * (d1 / nw)[:, None]), axis=1)
        out[sub] = d1 + 5 * d2
    return out


def _worst_region(F, W, region, pbi):
    s = np.zeros(len(W))
    np.add.at(s, region, pbi)
    return int(np.argmax(s))


def locate_worst(F, W, region, front, Z):
    crowd = np.bincount(region, minlength=len(W))
    phi = np.where(crowd == crowd.max())[0]
    pbi = _pbi(F, W, region, Z, np.isin(region, phi))
    ph = np.where(region == _worst_region(F, W, region, pbi))[0]
    R = ph[front[ph] == front[ph].max()]
    return int(R[int(np.argmax(pbi[R]))])


class MOEADD(LoopAlgorithm):
    """Steady-state decomposition-and-dominance MOEA: every solution lives in the subregion of its closest
    weight vector; each offspring insertion is followed by the removal of the worst solution of the most
    crowded subregion of the worst non-domination level."""

    def __init__(self, pop_size: int = 100, delta: float = 0.9, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta = float(delta)

    PER_STEP_OPTIMUM = False

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = np.argsort(pdist2(self.W, self.W), axis=1, kind="stable")[:, : self.T]
        return self.pop_size

    def _region(self, F):
        return np.argmax(1 - cosine_distance(F, self.W), axis=1)

    def start(self):
        F = objs(self.pop)
        self.region = self._region(F)
        self.front = nd_sort(F, None, np.inf)[0]
        self.Z = F.min(axis=0)

    def _cv(self, pop):
        C = cons(pop)
        return np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))

    def _victim(self, pop, F, CV):
        W, region, front, Z = self.W, self.region, self.front, self.Z
        if np.any(CV > 0):
            order = np.argsort(-CV, kind="stable")[: int(np.sum(CV > 0))]
            for s in order:
                if np.sum(region == region[s]) > 1:
                    return int(s)
            return int(order[0])
        if front.max() == 1:
            return locate_worst(F, W, region, front, Z)
        Fl = np.where(front == front.max())[0]
        if len(Fl) == 1:
            if np.sum(region == region[Fl[0]]) > 1:
                return int(Fl[0])
            return locate_worst(F, W, region, front, Z)
        sub = np.unique(region[Fl])
        crowd = np.bincount(region[np.isin(region, sub)], minlength=len(W))
        phi = np.where(crowd == crowd.max())[0]
        pbi = _pbi(F, W, region, Z, np.isin(region, phi))
        ph = np.where(region == _worst_region(F, W, region, pbi))[0]
        if len(ph) > 1:
            return int(ph[int(np.argmax(pbi[ph]))])
        return locate_worst(F, W, region, front, Z)

    def step(self):
        rng, N = self.rng, self.N
        for i in range(N):
            pop = self.pop
            Ei = np.where(np.isin(self.region, self.B[i]))[0]
            if rng.random() < self.delta and len(Ei) >= 2:
                P = Ei[tournament(2, 2, self._cv(pop[Ei]), rng=rng)]
            else:
                P = tournament(2, 2, self._cv(pop), rng=rng)
            off = self.evaluate(ga_half(self.problem, decs(pop[P]), rng=rng))
            pop = Population.merge(pop, off)
            F = objs(pop)
            self.region = np.concatenate([self.region, self._region(objs(off))])
            self.front = update_front(F, self.front)
            CV = self._cv(pop)
            self.Z = np.minimum(self.Z, objs(off)[0])
            x = self._victim(pop, F, CV)
            keep = np.delete(np.arange(len(pop)), x)
            pop = pop[keep]
            self.region = np.delete(self.region, x)
            self.front = update_front(objs(pop), self.front, x)
            self.pop = pop
        self._set_optimum()
