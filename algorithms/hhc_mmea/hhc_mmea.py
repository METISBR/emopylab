# emopylab 2026
"""HHC-MMEA (hybrid hierarchical clustering based multi-modal multi-objective evolutionary algorithm).

Reference:
Z. Ding, L. Cao, L. Chen, D. Sun, X. Zhang, and Z. Tao. Large-scale multimodal multiobjective
evolutionary optimization based on hybrid hierarchical clustering. Knowledge-Based Systems, 2023,
266: 110398.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, ga_half, kmeans, nd_sort, objs, tournament
from algorithms.community_utils.sparse_mask import probe_variables, ts
from algorithms.mp_mmea.mp_mmea import _update_gv
from core.population import Population

ALGORITHM_FLAGS = {'HHCMMEA': {'binary', 'integer', 'large', 'multi', 'multimodal', 'real', 'sparse'}}


def _hamming(A, B):
    return np.mean((A[:, None, :] > 0) != (B[None, :, :] > 0), axis=2)


def _select(pop, Dec, Mask, N, dis=None):
    F = objs(pop)
    PF = np.column_stack([F, dis]) if dis is not None else F
    N = min(N, len(pop))
    C = cons(pop)
    front, maxf = nd_sort(PF, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(PF, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], Dec[nxt], Mask[nxt], front[nxt], cd[nxt]


def _similarity(a, b):
    a, b = a > 0, b > 0
    with np.errstate(all="ignore"):
        return np.float64((a & b).sum()) / min(a.sum(), b.sum())


def _division_flag(Mask, front):
    mask = Mask[front == 1]
    dis = _hamming(mask, mask)
    M = dis.max()
    r, c = np.where(dis.T == M)          # first hit in column-major order
    l1, l2 = int(c[0]), int(r[0])
    s1 = int(np.where(np.all(Mask == mask[l1], axis=1))[0][0])
    s2 = int(np.where(np.all(Mask == mask[l2], axis=1))[0][0])
    sub1, sub2 = [s1], [s2]
    s = _similarity(mask[l1], mask[l2])
    flag = 0
    if M > 0.05 and s <= 0.5:
        flag = 1
        d1 = _hamming(mask[[l1]], Mask)[0]
        d2 = _hamming(mask[[l2]], Mask)[0]
        for i in range(len(Mask)):
            if i != s1 and i != s2:
                (sub1 if d1[i] <= d2[i] else sub2).append(i)
    if len(Mask) < 50:
        flag = 0
    return np.array(sub1), np.array(sub2), flag


def _sv(Mask, front):
    m = Mask[front == 1]
    with np.errstate(all="ignore"):
        return m.sum(0) / len(m)


class HHCMMEA(LoopAlgorithm):
    """Mask/real two-part encoding with per-variable scores from single-variable probes. Subpopulations are split every
    10 generations when their first-front masks form two dissimilar groups (and have at least 50 members), merged when
    their leading masks overlap by more than half, and refilled around the variables that the other subpopulations do not
    use; offspring masks are generated from the probe scores, or, once a subpopulation's guiding vector is decisive, from
    that guiding vector."""

    def _initialize_infill(self):
        rng, N, D = self.rng, self.N, self.D
        _, _, _, self.fitness = probe_variables(self)
        from operators.utility_functions.UniformPoint import UniformPoint
        P, _ = UniformPoint(N, D, "Latin", rng=rng)
        Dec = np.asarray(P) * (self.upper - self.lower) + self.lower
        Q, _ = UniformPoint(N, D, "Latin", rng=rng)
        Mask = (np.asarray(Q) > 0.5).astype(float)
        self._init = (Dec, Mask)
        return self.evaluate(Dec * Mask)

    def _initialize_advance(self, infills=None, **kwargs):
        Dec, Mask = self._init
        rng, N, D = self.rng, self.N, self.D
        idx = rng.permutation(N)
        p, d, m, f, c = _select(infills[idx], Dec[idx], Mask[idx], len(idx))
        self.pops, self.decs_, self.masks, self.front, self.crowd = [p], [d], [m], [f], [c]
        self.gv = [_update_gv(np.zeros(D), m, f)]
        self.leader = [None]
        self.pop = p
        self._set_optimum()

    def _sub_rank(self):
        allp = Population.merge(*self.pops)
        front, _ = nd_sort(objs(allp), None, np.inf)
        flag = np.concatenate([np.full(len(p), i) for i, p in enumerate(self.pops)])
        return np.array([front[flag == i].mean() for i in range(len(self.pops))])

    def _split_score(self, g, rng, thresh_on_fgv=False):
        fgv = g.copy()
        fgv[fgv < 1e-2] = 0
        lab = kmeans(fgv[:, None], 2, rng)
        src = fgv if thresh_on_fgv else g
        with np.errstate(all="ignore"):
            s1 = src[lab == 0].sum() / (lab == 0).sum()
            s2 = src[lab == 1].sum() / (lab == 1).sum()
        return fgv, lab, s1, s2, np.fmax(s1, s2)

    def _gaoff(self, pdec):
        enc = self.encoding
        if np.any(enc != 4):
            od = ga_half(self.problem, pdec, rng=self.rng)
            od[:, enc == 4] = 1
            return od
        return np.ones((len(pdec) // 2, self.D))

    def _operator1(self, pdec, pmask):
        rng, fit = self.rng, self.fitness
        h = len(pdec) // 2
        P1, P2 = pmask[:h] > 0, pmask[h:] > 0
        off = P1.astype(float)
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where(P1[i] & ~P2[i])[0]
                t = ts(-fit[idx], rng)
                if t is not None:
                    off[i, idx[t]] = 0
            else:
                idx = np.where(~P1[i] & P2[i])[0]
                t = ts(fit[idx], rng)
                if t is not None:
                    off[i, idx[t]] = 1
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where(off[i] > 0)[0]
                t = ts(-fit[idx], rng)
                if t is not None:
                    off[i, idx[t]] = 0
            else:
                idx = np.where(off[i] == 0)[0]
                t = ts(fit[idx], rng)
                if t is not None:
                    off[i, idx[t]] = 1
        return self._gaoff(pdec), off

    def _operator2(self, pdec, pmask, score, delta):
        rng, D = self.rng, self.D
        h = len(pdec) // 2
        P1, P2 = pmask[:h] > 0, pmask[h:] > 0
        off = P1.astype(float)
        for i in range(h):
            i1, i2 = np.where(P1[i] & ~P2[i])[0], np.where(~P1[i] & P2[i])[0]
            p1, p2 = 1 / (1 + np.exp(-score[i1])), 1 / (1 + np.exp(-score[i2]))
            off[i, i1[p1 < rng.random(len(p1))]] = 0
            off[i, i2[p2 > rng.random(len(p2))]] = 1
        if rng.random() < 1 - delta:
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
        else:
            ex = rng.random((h, D)) < 1.0 / D
            for i in range(h):
                sub = np.where(ex[i])[0]
                if len(sub):
                    on = off[i, sub] > 0
                    rate = np.where(on, 1 - score[sub], score[sub])
                    flip = sub[rng.random(len(sub)) < rate]
                    off[i, flip] = 1 - off[i, flip]
        return ga_half(self.problem, pdec, rng=rng), off

    def _leader(self, Mask, front, g):
        fs = (g > self.rng.random(self.D)).astype(float)
        mask = Mask[front == 1]
        d = _hamming(mask, fs[None])[:, 0]
        return mask[int(np.where(d == d.min())[0][0])]

    def step(self):
        rng, N, D = self.rng, self.N, self.D
        K = len(self.pops)
        rank = np.argsort(self._sub_rank(), kind="stable")
        for i in range(K):
            r = rank[i]
            self.gv[r] = _update_gv(self.gv[r], self.masks[r], self.front[r])
            mate = tournament(2, 2 * len(self.pops[r]), self.front[r], -self.crowd[r], rng=rng)
            _, _, _, _, s = self._split_score(self.gv[r], rng)
            if s < 0.5 or K == 1:
                od, om = self._operator1(self.decs_[r][mate], self.masks[r][mate])
            else:
                self.leader[r] = self._leader(self.masks[r], self.front[r], self.gv[r])
                od, om = self._operator2(self.decs_[r][mate], self.masks[r][mate], self.gv[r], self.FE / self.max_FE)
            off = self.evaluate(od * om)
            self.pops[r] = Population.merge(self.pops[r], off)
            self.decs_[r] = np.vstack([self.decs_[r], od])
            self.masks[r] = np.vstack([self.masks[r], om])
            half = len(self.pops[r]) // 2
            if K == 1:
                out = _select(self.pops[r], self.decs_[r], self.masks[r], half)
            else:
                R = np.zeros(D)
                for j in range(K):
                    if j != i:
                        fgv, lab, s1, s2, s = self._split_score(self.gv[rank[j]], rng, thresh_on_fgv=True)
                        if s > 0.5:
                            sel = lab == (0 if s1 > s2 else 1)
                            R[sel] += fgv[sel]
                R[R > 0] = 1
                dis = np.sum((R > 0)[None, :] & (self.masks[r] > 0), axis=1)
                out = _select(self.pops[r], self.decs_[r], self.masks[r], half, dis)
            self.pops[r], self.decs_[r], self.masks[r], self.front[r], self.crowd[r] = out
        if int(np.ceil(self.FE / N)) % 10 == 0:
            self._restructure()
        self.pop = Population.merge(*self.pops)

    def _restructure(self):
        rng, N, D = self.rng, self.N, self.D
        K = len(self.pops)
        k = K
        while True:
            flag = 0
            for i in range(K):
                row, col, flag = _division_flag(self.masks[i], self.front[i])
                if flag == 1:
                    k += 1
                    p, m, d, f = self.pops[i], self.masks[i], self.decs_[i], self.front[i]
                    self.pops.append(p[col]); self.masks.append(m[col]); self.decs_.append(d[col]); self.front.append(f[col])
                    self.gv.append(_sv(m[col], f[col])); self.crowd.append(np.zeros(len(col)))
                    self.pops[i], self.masks[i], self.decs_[i], self.front[i] = p[row], m[row], d[row], f[row]
                    self.gv[i] = _sv(m[row], f[row])
                    while len(self.leader) < k:
                        self.leader.append(None)
                    self.leader[i], self.leader[k - 1] = self.masks[i][0], self.masks[k - 1][0]
            K = k
            if flag == 0:
                break
        refs = [self.leader[m] if self.leader[m] is not None else self.masks[m][0] for m in range(K)]
        ss, index = self._similar(refs)
        while K > 2 and ss > 0.5:
            i, j = index
            out = _select(Population.merge(self.pops[i], self.pops[j]), np.vstack([self.decs_[i], self.decs_[j]]),
                          np.vstack([self.masks[i], self.masks[j]]), len(self.pops[i]))
            self.pops[i], self.decs_[i], self.masks[i], self.front[i], self.crowd[i] = out
            for lst in (self.pops, self.decs_, self.masks, self.front, self.gv):
                del lst[j]
            K -= 1
            ss, index = self._similar([self.masks[m] for m in range(K)])
        popsize = N // K
        rank = np.argsort(self._sub_rank(), kind="stable")
        from operators.utility_functions.UniformPoint import UniformPoint
        for i in range(K):
            r = rank[i]
            ln = popsize - len(self.pops[r])
            if ln > 0:
                P, _ = UniformPoint(ln, D, "Latin", rng=rng)
                dec = np.asarray(P) * (self.lower + rng.random((ln, D)) * (self.upper - self.lower))
                mask = np.zeros((ln, D))
                Fg = np.zeros(D)
                for j in range(K):
                    if j != i:
                        Fg = Fg + self.gv[rank[j]]
                for j in range(ln):
                    nsel = int(np.floor(rng.random() * D))
                    if nsel:
                        mask[j, tournament(2, nsel, Fg, rng=rng)] = 1
                self.pops[r] = Population.merge(self.pops[r], self.evaluate(dec * mask))
                self.masks[r] = np.vstack([self.masks[r], mask])
                self.decs_[r] = np.vstack([self.decs_[r], dec])
            out = _select(self.pops[r], self.decs_[r], self.masks[r], popsize)
            self.pops[r], self.decs_[r], self.masks[r], self.front[r], self.crowd[r] = out
        del self.crowd[K:]

    @staticmethod
    def _similar(fs):
        ss, index = 0.0, None
        for i in range(len(fs) - 1):
            for j in range(i + 1, len(fs)):
                a = fs[i][0] if np.ndim(fs[i]) == 2 else fs[i]
                b = fs[j][0] if np.ndim(fs[j]) == 2 else fs[j]
                v = _similarity(a, b)
                if v > ss:
                    ss, index = v, (i, j)
        return ss, index
