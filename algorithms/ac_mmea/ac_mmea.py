# emopylab 2026
"""AC-MMEA (adaptive merging and coordinated offspring generation based multi-modal multi-objective evolutionary algorithm).

Reference:
X. Wang, T. Zheng, and Y. Jin. Adaptive merging and coordinated offspring generation in multi-
population evolutionary multi-modal multi-objective optimization. Proceedings of the International
Conference on Data-driven Optimization of Complex Systems, 2023.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import spea
from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, ga_half, nd_sort, objs, tournament
from algorithms.community_utils.sparse_mask import ts
from algorithms.mp_mmea.mp_mmea import _update_gv
from core.population import Population

ALGORITHM_FLAGS = {'ACMMEA': {'integer', 'large', 'multi', 'multimodal', 'real', 'sparse'}}


def _score(gv, D):
    """Guiding vector as the operator reads it: after merges it is a stacked matrix and indexing is column-major."""
    return np.atleast_2d(gv).ravel(order="F")[:D]


def _gv_update(gv, Mask, front):
    gv = np.atleast_2d(gv)
    v = _update_gv(np.zeros(Mask.shape[1]), Mask, front)
    return v[None] if np.all(gv == 0) else 0.9 * gv + 0.1 * v


def clustering(X, n, bound, dim):
    """Agglomerative clustering (single linkage to cluster representatives) with at most ``n`` members per cluster; the
    merge radius starts at ``sqrt(dim) * |bound[1] - bound[0]|``."""
    a = len(X)
    G = [[i] for i in range(a)]
    reps = list(range(a))
    M = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    found = True
    while found:
        found = False
        md = np.sqrt(dim * (bound[1] - bound[0]) ** 2)
        r = s = None
        for i in range(len(G) - 1):
            for j in range(i + 1, len(G)):
                if len(G[i]) + len(G[j]) < n + 1 and md >= M[i, j]:
                    md, r, s, found = M[i, j], i, j, True
        if r is not None:
            G[r] = G[r] + G[s]
            reps = [p for p in reps if p not in set(G[s])]
            del G[s]
            M = np.delete(np.delete(M, s, 0), s, 1)
            h = X[reps]
            Dm = np.sqrt(((X[G[r]][:, None] - h[None]) ** 2).sum(-1)).min(axis=0)
            M[:, r], M[r, :] = Dm, Dm
            if not any(len(g) == 1 for g in G):
                found = False
    return G


def cpu_group(n_groups, ratio, D):
    per = D // n_groups
    if per == 1:
        return np.arange(1, D + 1), D
    INDEX = np.concatenate([np.repeat(np.arange(1, n_groups + 1), per), np.full(D - per * n_groups, n_groups + 1)])
    idx = np.empty(D)
    idx[np.argsort(ratio, kind="stable")] = INDEX
    return idx, n_groups if D % n_groups == 0 else n_groups + 1


def _pm(off, lo, up, mask, rng, disM):
    N, D = off.shape
    mu = rng.random((N, D))
    L, U = np.broadcast_to(lo, off.shape), np.broadcast_to(up, off.shape)
    off = np.minimum(np.maximum(off, L), U)
    with np.errstate(all="ignore"):
        t = mask & (mu <= 0.5)
        off[t] = off[t] + (U[t] - L[t]) * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - L[t]) / (U[t] - L[t])) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = mask & (mu > 0.5)
        off[t] = off[t] + (U[t] - L[t]) * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (U[t] - off[t]) / (U[t] - L[t])) ** (disM + 1)) ** (1 / (disM + 1)))
    return off


def _select(pop, Dec, Mask, N, dis=None):
    F = objs(pop)
    PF = np.column_stack([F, dis]) if dis is not None else F
    uni = np.unique(PF, axis=0, return_index=True)[1]
    PF, pop, Dec, Mask = PF[uni], pop[uni], Dec[uni], Mask[uni]
    N = min(N, len(pop))
    C = cons(pop)
    front, maxf = nd_sort(PF, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(PF, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], Dec[nxt], Mask[nxt], front[nxt], cd[nxt]


def _select_s(pop, Dec, Mask, N):
    F = objs(pop)
    uni = np.unique(F, axis=0, return_index=True)[1]
    F, pop, Dec, Mask = F[uni], pop[uni], Dec[uni], Mask[uni]
    N = min(N, len(pop))
    fit = spea.cal_fitness(F)
    C = cons(pop)
    front, _ = nd_sort(F, C if C.size else None, N)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        tmp = np.where(nxt)[0]
        nxt[tmp[spea.truncation(F[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt], Dec[nxt], Mask[nxt], front[nxt], fit[nxt]


def _similarity(a, b):
    a, b = a > 0, b > 0
    with np.errstate(all="ignore"):
        return np.float64((a & b).sum()) / min(a.sum(), b.sum())


class ACMMEA(LoopAlgorithm):
    """Subpopulations are found by size-limited agglomerative clustering of the initial decision vectors. In the first 30% of
    the budget they evolve under NSGA-II selection, worse ranked ones being pushed away from the best masks of better ones;
    afterwards SPEA2 selection is used. Subpopulations whose best masks overlap by more than 70% are merged every 50
    generations. Real variables are mutated with group-wise rates that depend on how often the variable is active."""

    def _initialize_infill(self):
        rng, N, D = self.rng, self.N, self.D
        Dec = self.lower + rng.random((N, D)) * (self.upper - self.lower)
        Mask = np.zeros((N, D))
        gv = np.ones(D)
        for i in range(N):
            Mask[i, tournament(2, int(np.ceil(rng.random() * D)), gv, rng=rng)] = 1
            gv[Mask[i] == 1] += 1
        self._init = (Dec, Mask)
        return self.evaluate(Dec * Mask)

    def _initialize_advance(self, infills=None, **kwargs):
        Dec, Mask = self._init
        D = self.D
        bound = np.concatenate([self.lower, self.upper])
        slst = clustering(Dec * Mask, 20, bound, D)
        self.pops, self.decs_, self.masks, self.front, self.crowd, self.gv, self.fit = [], [], [], [], [], [], []
        for g in slst:
            g = np.array(g)
            p, d, m, f, c = _select(infills[g], Dec[g], Mask[g], len(g))
            self.pops.append(p); self.decs_.append(d); self.masks.append(m); self.front.append(f); self.crowd.append(c)
            self.gv.append(_gv_update(np.zeros(D), m, f)); self.fit.append(None)
        self.stage1, self.timea, self.rank = 1, 0, np.arange(len(self.pops))
        self.pop = Population.merge(*self.pops)
        self._set_optimum()

    def _sub_rank(self):
        allp = Population.merge(*self.pops)
        front, _ = nd_sort(objs(allp), None, np.inf)
        flag = np.concatenate([np.full(len(p), i) for i, p in enumerate(self.pops)])
        return np.array([front[flag == i].mean() for i in range(len(self.pops))])

    def _operator(self, pdec, pmask, gv, stage1):
        rng, D = self.rng, self.D
        score = _score(gv, D)
        N = len(pdec)
        h = N // 2
        P1, P2 = pmask[:h] > 0, pmask[h:] > 0
        off = P1.astype(float)
        for i in range(h):
            i1, i2 = np.where(P1[i] & ~P2[i])[0], np.where(~P1[i] & P2[i])[0]
            p1, p2 = 1 / (1 + np.exp(-score[i1])), 1 / (1 + np.exp(-score[i2]))
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
        grp, MAX = cpu_group(5, off.sum(0) / N, D)
        return self._ga_smm(pdec, grp, MAX, stage1), off

    def _ga_smm(self, P, grp, MAX, flag, proC=1.0, disC=20.0, proM=1.0, disM=0.5):
        rng, D = self.rng, self.D
        enc = self.encoding
        out = ga_half(self.problem, P, (proC, disC, proM, disM), rng=rng)   # binary / label / permutation parts
        real = np.isin(enc, (1, 2))
        if not real.any():
            return out
        h = len(P) // 2
        A, B = P[:h][:, real], P[h:2 * h][:, real]
        n, d = A.shape
        mu = rng.random((n, d))
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (disC + 1)), (2 - 2 * mu) ** (-1 / (disC + 1)))
        beta = beta * (-1.0) ** rng.integers(0, 2, (n, d))
        beta[rng.random((n, d)) < 0.5] = 1
        beta[np.repeat(rng.random((n, 1)) > proC, d, axis=1)] = 1
        off = (A + B) / 2 + beta * (A - B) / 2
        lo, up, g = self.lower[real], self.upper[real], grp[real]
        rates = (4, 4, 4) if flag else (4, 2, 0.25)
        sets = (g == MAX, (g == MAX - 2) | (g == MAX - 1), (g == MAX - 3) | (g == MAX - 4))
        for rate, sel in zip(rates, sets):
            site = rng.random((n, d)) < rate * proM / D
            off = _pm(off, lo, up, site & sel[None, :], rng, disM)
        out[:, real] = off
        return out

    def _merge_similar(self):
        K = len(self.pops)
        fs = [int(np.argmin(objs(p).mean(1))) for p in self.pops]
        flag = np.zeros(K, int)
        for i in range(K - 1):
            if flag[i] == 0 or flag[i] == i + 1:
                for j in range(i + 1, K):
                    if flag[j] == 0 and _similarity(self.masks[i][fs[i]], self.masks[j][fs[j]]) > 0.7:
                        flag[i], flag[j] = i + 1, i + 1
        keep = [i for i in range(K) if flag[i] == 0]
        groups = [np.where(flag == k)[0] for k in np.unique(flag) if k != 0]
        P, Dd, Mm, G = [], [], [], []
        for i in keep:
            P.append(self.pops[i]); Dd.append(self.decs_[i]); Mm.append(self.masks[i]); G.append(self.gv[i])
        for c in groups:
            P.append(Population.merge(*[self.pops[i] for i in c])); Dd.append(np.vstack([self.decs_[i] for i in c]))
            Mm.append(np.vstack([self.masks[i] for i in c])); G.append(np.vstack([np.atleast_2d(self.gv[i]) for i in c]))
        self.pops, self.decs_, self.masks, self.gv = P, Dd, Mm, G
        K = len(P)
        self.front, self.crowd, self.fit = [None] * K, [None] * K, [None] * K
        for i in range(K):
            self.pops[i], self.decs_[i], self.masks[i], self.front[i], self.crowd[i] = _select(self.pops[i], self.decs_[i], self.masks[i], self.N // K)

    def _add(self, r, od, om):
        self.pops[r] = Population.merge(self.pops[r], self.evaluate(od * om))
        self.decs_[r] = np.vstack([self.decs_[r], od])
        self.masks[r] = np.vstack([self.masks[r], om])

    def step(self):
        rng, N, D = self.rng, self.N, self.D
        K = len(self.pops)
        if self.FE < 0.3 * self.max_FE:
            self.rank = rank = np.argsort(self._sub_rank(), kind="stable")
            fs = {}
            for i in range(K):
                r = rank[i]
                self.gv[r] = _gv_update(self.gv[r], self.masks[r], self.front[r])
                mate = tournament(2, 2 * len(self.pops[r]), self.front[r], -self.crowd[r], rng=rng)
                self._add(r, *self._operator(self.decs_[r][mate], self.masks[r][mate], self.gv[r], self.stage1))
                if i > 0:
                    R = np.zeros(D)
                    for j in range(i):
                        fs[rank[j]] = int(np.argmin(objs(self.pops[rank[j]]).mean(1)))
                        R = R + self.masks[rank[j]][fs[rank[j]]]
                    dis = np.sum((R > 0)[None, :] & (self.masks[r] > 0), axis=1)
                    out = _select(self.pops[r], self.decs_[r], self.masks[r], N // K, dis)
                else:
                    out = _select(self.pops[r], self.decs_[r], self.masks[r], N // K)
                self.pops[r], self.decs_[r], self.masks[r], self.front[r], self.crowd[r] = out
            if int(np.ceil(self.FE / N)) % 50 == 0:
                self._merge_similar()
        else:
            if np.any(self.rank >= K) or len(self.rank) != K:
                self.rank = np.argsort(self._sub_rank(), kind="stable")   # stale ranking after a stage-I merge
            rank = self.rank
            if self.timea == 0:
                for i in range(K):
                    r = rank[i]
                    self.pops[r], self.decs_[r], self.masks[r], self.front[r], self.fit[r] = _select_s(self.pops[r], self.decs_[r], self.masks[r], len(self.pops[r]))
                    self.gv[r] = _gv_update(np.zeros(D), self.masks[r], self.front[r])
            self.timea += 1
            self.stage1 = 0
            for i in range(K):
                r = rank[i]
                self.gv[r] = _gv_update(self.gv[r], self.masks[r], self.front[r])
                mate = tournament(2, 2 * len(self.pops[r]), self.front[r], self.fit[r], rng=rng)
                self._add(r, *self._operator(self.decs_[r][mate], self.masks[r][mate], self.gv[r], 0))
                self.pops[r], self.decs_[r], self.masks[r], self.front[r], self.fit[r] = _select_s(self.pops[r], self.decs_[r], self.masks[r], N // K)
            if int(np.ceil(self.FE / N)) % 50 == 0:
                self._merge_similar()
                self.timea = 0
                self.rank = np.argsort(self._sub_rank(), kind="stable")
        self.pop = Population.merge(*self.pops)
