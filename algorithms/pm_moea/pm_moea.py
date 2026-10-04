# emopylab 2026
"""PM-MOEA (pattern mining based multi-objective evolutionary algorithm).

Reference:
Y. Tian, C. Lu, X. Zhang, F. Cheng, and Y. Jin. A pattern mining based evolutionary algorithm for
large-scale sparse multi-objective optimization problems. IEEE Transactions on Cybernetics, 2022,
52(7): 6784-6797.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, nd_sort, objs, pdist2, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'PMMOEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


def _truncation(X, K):
    d = pdist2(X, X)
    np.fill_diagonal(d, np.inf)
    return truncate_lexi(d, K)


def binary_crossover(P1, P2, rng):
    off = P1.astype(bool).copy()
    for i in range(len(off)):
        diff = np.where(P1[i] != P2[i])[0]
        if len(diff) == 0:
            continue
        one = off[i, diff].mean()
        r = min(0.5, 2 * one, 2 * (1 - one))
        with np.errstate(all="ignore"):
            rate = np.where(off[i, diff], r / 2 / one, r / 2 / (1 - one))
        ex = rng.random(len(diff)) < rate
        off[i, diff[ex]] = ~off[i, diff[ex]]
    return off


def binary_mutation(off, rng):
    N, D = off.shape
    one = off.mean(axis=1, keepdims=True)
    r = np.minimum(np.minimum(1.0 / D, 2 * one), 2 * (1 - one))
    with np.errstate(all="ignore"):
        rate = np.where(off, r / 2 / one, r / 2 / (1 - one))
    ex = rng.random((N, D)) < rate
    off = off.copy()
    off[ex] = ~off[ex]
    return off


def _cal_obj(P, T, lens, maximize):
    n = len(P)
    out = np.ones((n, 2))
    ln = P.sum(axis=1)
    if maximize:
        Tx = np.all(P[:, None, :] >= T[None, :, :], axis=2)
    else:
        Tx = ~np.any(P[:, None, :] & ~T[None, :, :], axis=2)
    for i in range(n):
        if not Tx[i].any():
            continue
        out[i, 0] = 1 - Tx[i].mean()
        with np.errstate(all="ignore"):
            out[i, 1] = 1 - (lens[Tx[i]].mean() / ln[i] if maximize else np.mean(ln[i] / lens[Tx[i]]))
    return out


def _pattern_selection(P, F, N):
    _, uni = np.unique(P.astype(np.uint8), axis=0, return_index=True)
    uni = np.sort(uni)                                            # 'stable': first occurrences in original order
    P, F = P[uni], F[uni]
    front, maxf = nd_sort(F, None, N)
    nxt = front <= maxf
    last = np.where(front == maxf)[0]
    K = int(nxt.sum()) - N
    if K > 0:
        nxt[last[_truncation(P[last].astype(float), K)]] = False
    return P[nxt], F[nxt], front[nxt]


def _mining(T, prev, N, maximize, rng):
    ns = len(T)
    pat = np.zeros((N, prev.shape[1]), bool)
    for i in range(N):
        rows = rng.permutation(ns)[: int(np.ceil(rng.random() ** 2 * ns))]
        pat[i] = T[rows].any(axis=0) if maximize else T[rows].all(axis=0)
    eye = np.eye(prev.shape[1], dtype=bool)
    P = np.vstack([prev, pat, (~eye if maximize else eye), T])
    lens = T.sum(axis=1)
    F = _cal_obj(P, T, lens, maximize)
    P, F, front = _pattern_selection(P, F, N)
    for _ in range(10):
        pool = tournament(2, 2 * N, front, rng=rng)
        off = binary_crossover(P[pool[0::2]], P[pool[1::2]], rng)
        off = binary_mutation(off, rng)
        P, F, front = _pattern_selection(np.vstack([P, off]), np.vstack([F, _cal_obj(off, T, lens, maximize)]), N)
    return P


def pos_mining(Mask, MaxP, MinP, N, rng):
    nz = Mask.any(axis=0)
    out = []
    for prev, mx in ((MaxP, True), (MinP, False)):
        p = _mining(Mask[:, nz], prev[:, nz], N, mx, rng)
        new = np.zeros((len(p), Mask.shape[1]), bool)
        new[:, nz] = p
        r = min(len(new), len(prev))
        new[:r, ~nz] = prev[:r][:, ~nz]
        out.append(new)
    return out[0], out[1], np.where(nz)[0]


class PMMOEA(LoopAlgorithm):
    """Pattern-mining MOEA for sparse problems: the frequent subsets (maximal patterns) and rare subsets (minimal
    patterns) of the non-dominated masks are mined and used to steer the crossover of the masks."""

    def _initialize_infill(self):
        N, D, rng = self.N, self.D, self.rng
        lo, up, enc = self.lower, self.upper, self.encoding
        Dec = lo + rng.random((N, D)) * (up - lo)
        Dec[:, enc == 4] = 1
        Mask = np.zeros((N, D), bool)
        for i in range(N):
            Mask[i, rng.permutation(D)[: int(np.ceil(rng.random() ** 2 * D))]] = True
        pop = self.evaluate(Dec * Mask)
        self.pop_, self.Dec, self.Mask, self.front = self._select(pop, Dec, Mask, N)
        self.MaxP = np.zeros((20, D), bool)
        self.MinP = self.MaxP.copy()
        return self.pop_

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _select(self, pop, Dec, Mask, N):
        C = cons(pop)
        F = objs(pop)
        front, maxf = nd_sort(F, C if C.size else None, N)
        nxt = front <= maxf
        last = np.where(front == maxf)[0]
        K = int(nxt.sum()) - N
        if K > 0:
            nxt[last[_truncation(F[last], K)]] = False
        return pop[nxt], Dec[nxt], Mask[nxt], front[nxt]

    def _operator(self, ParentDec, ParentMask, MaxP, MinP, nonzero, PopDec):
        rng, D = self.rng, self.D
        n = len(ParentMask)
        h = n // 2
        P1, P2 = ParentMask[:h], ParentMask[h:]
        exist = np.zeros((len(MaxP), len(MinP)), bool)
        for i in range(len(MinP)):
            exist[:, i] = np.all(MaxP[:, MinP[i]], axis=1)
        MaxP = MaxP[exist.any(axis=1)]
        MinP = MinP[exist.any(axis=0)]
        if len(MaxP) and len(MinP):
            Off = np.zeros((h, D), bool)
            for i in range(h):
                maxp = MaxP[int(rng.integers(len(MaxP)))]
                minp = MinP[int(rng.integers(len(MinP)))]
                maxp = maxp & ~minp
                cols = nonzero[maxp]
                Off[i, cols] = binary_crossover(P1[i:i + 1, cols], P2[i:i + 1, cols], rng)[0]
                Off[i, nonzero[minp]] = True
        else:
            Off = binary_crossover(P1, P2, rng)
        Off = binary_mutation(Off, rng)
        enc = self.encoding
        if np.any(enc != 4):
            OffDec = ga_half(self.problem, ParentDec, rng=rng)
            OffDec[:, enc == 4] = 1
        else:
            OffDec = np.ones(Off.shape)
        prod = OffDec * Off
        _, uni = np.unique(prod, axis=0, return_index=True)
        OffDec, Off, prod = OffDec[uni], Off[uni], prod[uni]
        have = {tuple(r) for r in PopDec}
        keep = np.array([tuple(r) not in have for r in prod], dtype=bool)
        return OffDec[keep], Off[keep]

    def step(self):
        N, rng = self.N, self.rng
        first = self.front == 1
        self.MaxP, self.MinP, nonzero = pos_mining(decs(self.pop_[first]) != 0, self.MaxP, self.MinP, 20, rng)
        pool = tournament(2, N, self.front, rng=rng)
        OffDec, OffMask = self._operator(self.Dec[pool], self.Mask[pool], self.MaxP[:, nonzero], self.MinP[:, nonzero], nonzero, decs(self.pop_))
        if len(OffDec):
            off = self.evaluate(OffDec * OffMask)
            self.pop_, self.Dec, self.Mask, self.front = self._select(
                Population.merge(self.pop_, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), N)
        self.pop = self.pop_
