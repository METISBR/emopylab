# emopylab 2026
"""MP-MMEA (multi-population multi-modal multi-objective evolutionary algorithm).

Reference:
Y. Tian, R. Liu, X. Zhang, H. Ma, K. C. Tan, and Y. Jin. A multipopulation evolutionary algorithm
for solving large-scale multimodal multiobjective optimization problems. IEEE Transactions on
Evolutionary Computation, 2021, 25(3): 405-418.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, ga_half, nd_sort, objs, tournament
from algorithms.community_utils.sparse_mask import ts
from core.population import Population

ALGORITHM_FLAGS = {'MPMMEA': {'integer', 'large', 'multi', 'multimodal', 'real', 'sparse'}}


def _select(pop, Dec, Mask, N, dis=None):
    F = objs(pop)
    PopObj = np.column_stack([F, dis]) if dis is not None else F
    uni = np.unique(PopObj, axis=0, return_index=True)[1]
    PopObj, pop, Dec, Mask = PopObj[uni], pop[uni], Dec[uni], Mask[uni]
    N = min(N, len(pop))
    C = cons(pop)
    front, maxf = nd_sort(PopObj, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(PopObj, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], Dec[nxt], Mask[nxt], front[nxt], cd[nxt]


def _update_gv(gv, Mask, front):
    M = (Mask[front == 1] > 0)
    n = len(M)
    k = int(np.ceil(0.1 * n))
    v = np.zeros(M.shape[1])
    Mf = M.astype(float)
    ham = np.mean(Mf[:, None, :] != Mf[None, :, :], axis=2)
    half = Mf * 0.5
    for i in range(n):
        near = np.argsort(ham[i], kind="stable")[:k]
        v += (half[i][None, :] + half[near]).sum(axis=0) / (n * k)
    return v if np.all(gv == 0) else 0.9 * gv + 0.1 * v


class MPMMEA(LoopAlgorithm):
    """Several subpopulations each keep a different Pareto-optimal subset of the sparse variables: better ranked
    subpopulations are selected freely, worse ones are additionally pushed away from the best masks of the better ones; every
    50 generations similar subpopulations are merged, or (if every subpopulation reaches the first front) a new one is added."""

    def _initialize_infill(self):
        rng, N, D = self.rng, self.N, self.D
        Dec = self.lower + rng.random((N, D)) * (self.upper - self.lower)
        Mask = np.zeros((N, D))
        gv = np.ones(D)
        for i in range(N):
            Mask[i, tournament(2, int(np.ceil(rng.random() * D)), gv, rng=rng)] = 1
            gv[Mask[i] == 1] += 1
        self._init_dec, self._init_mask = Dec, Mask
        return self.evaluate(Dec * Mask)

    def _initialize_advance(self, infills=None, **kwargs):
        pop, Dec, Mask = infills, self._init_dec, self._init_mask
        rng, N, D = self.rng, self.N, self.D
        K = 2
        n = N // K
        index = rng.permutation(n * K)
        self.pops, self.decs_, self.masks, self.gv, self.front, self.crowd = [], [], [], [], [], []
        for i in range(K):
            sel = index[i::K]
            p, d, m, f, c = _select(pop[sel], Dec[sel], Mask[sel], len(sel))
            self.pops.append(p), self.decs_.append(d), self.masks.append(m), self.front.append(f), self.crowd.append(c)
            self.gv.append(_update_gv(np.zeros(D), m, f))
        self.pop = Population.merge(*self.pops)
        self._set_optimum()

    def _sub_rank(self):
        allp = Population.merge(*self.pops)
        front, _ = nd_sort(objs(allp), None, np.inf)
        flag = np.concatenate([np.full(len(p), i) for i, p in enumerate(self.pops)])
        return (np.array([front[flag == i].mean() for i in range(len(self.pops))]),
                np.array([front[flag == i].min() for i in range(len(self.pops))]))

    def _operator(self, pdec, pmask, score):
        rng = self.rng
        h = len(pdec) // 2
        P1, P2 = pmask[:h] > 0, pmask[h:] > 0
        off = P1.astype(float)
        for i in range(h):
            i1 = np.where(P1[i] & ~P2[i])[0]
            i2 = np.where(~P1[i] & P2[i])[0]
            p1 = 1 / (1 + np.exp(-score[i1]))
            p2 = 1 / (1 + np.exp(-score[i2]))
            off[i, i1[p1 < rng.random(len(p1))]] = 0
            off[i, i2[p2 > rng.random(len(p2))]] = 1
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where(off[i] > 0)[0]
                t = ts(score[idx], rng)
                if t is not None:
                    off[i, idx[t]] = 0
            else:
                idx = np.where(off[i] == 0)[0]
                t = ts(-score[idx], rng)
                if t is not None:
                    off[i, idx[t]] = 1
        return ga_half(self.problem, pdec, rng=rng), off

    def step(self):
        rng, N, D = self.rng, self.N, self.D
        K = len(self.pops)
        rank = np.argsort(self._sub_rank()[0], kind="stable")
        fs = {}
        for i in range(K):
            r = rank[i]
            self.gv[r] = _update_gv(self.gv[r], self.masks[r], self.front[r])
            mate = tournament(2, 2 * len(self.pops[r]), self.front[r], -self.crowd[r], rng=rng)
            od, om = self._operator(self.decs_[r][mate], self.masks[r][mate], self.gv[r])
            off = self.evaluate(od * om)
            self.pops[r] = Population.merge(self.pops[r], off)
            self.decs_[r] = np.vstack([self.decs_[r], od])
            self.masks[r] = np.vstack([self.masks[r], om])
            if i > 0:
                for j in range(i):
                    fs[rank[j]] = int(np.argmin(objs(self.pops[rank[j]]).mean(axis=1)))
                R = np.zeros(D)
                for j in range(i):
                    R = R + self.masks[rank[j]][fs[rank[j]]]
                R[R > 0] = 1
                dis = np.sum((R > 0)[None, :] & (self.masks[r] > 0), axis=1)
                out = _select(self.pops[r], self.decs_[r], self.masks[r], N // K, dis)
            else:
                out = _select(self.pops[r], self.decs_[r], self.masks[r], N // K)
            self.pops[r], self.decs_[r], self.masks[r], self.front[r], self.crowd[r] = out
        if int(np.ceil(self.FE / N)) % 50 == 0:
            best = self._sub_rank()[1]
            division = bool(np.all(best == 1))
            K = len(self.pops)
            fsx = [int(np.argmin(objs(p).mean(axis=1))) for p in self.pops]
            ss, pair = 0.0, None
            for i in range(K - 1):
                for j in range(i + 1, K):
                    a, b = self.masks[i][fsx[i]] > 0, self.masks[j][fsx[j]] > 0
                    with np.errstate(all="ignore"):
                        s = np.sum(a & b) / min(np.sum(a), np.sum(b))
                    if s > ss:
                        ss, pair = s, (i, j)
            if ss > 0.5:
                K -= 1
                i, j = pair
                out = _select(Population.merge(self.pops[i], self.pops[j]), np.vstack([self.decs_[i], self.decs_[j]]),
                              np.vstack([self.masks[i], self.masks[j]]), N // K)
                self.pops[i], self.decs_[i], self.masks[i], self.front[i], self.crowd[i] = out
                for lst in (self.pops, self.decs_, self.masks, self.gv, self.front, self.crowd):
                    del lst[j]
            elif division:
                for i in range(K):
                    out = _select(self.pops[i], self.decs_[i], self.masks[i], N // (K + 1))
                    self.pops[i], self.decs_[i], self.masks[i], self.front[i], self.crowd[i] = out
                K += 1
                n = N // K
                Dec = self.lower + rng.random((n, D)) * (self.upper - self.lower)
                Mask = np.zeros((n, D))
                F = np.sum(self.gv[: K - 1], axis=0)
                for i in range(n):
                    cnt = int(np.floor(rng.random() * D))
                    if cnt > 0:
                        Mask[i, tournament(2, cnt, F, rng=rng)] = 1
                p = self.evaluate(Dec * Mask)
                p, d, m, f, c = _select(p, Dec, Mask, len(p))
                self.pops.append(p), self.decs_.append(d), self.masks.append(m), self.front.append(f), self.crowd.append(c)
                self.gv.append(_update_gv(np.zeros(D), m, f))
        self.pop = Population.merge(*self.pops)
