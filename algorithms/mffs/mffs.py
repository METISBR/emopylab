# emopylab 2026
"""MFFS (multiform feature selection).

Reference:
R. Jiao, B. Xue, and M. Zhang. Benefiting from single-objective feature selection to multiobjective
feature selection: A multiform approach. IEEE Transactions on Cybernetics, 2023, 53(12): 7773-7786.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MFFS': {'binary', 'multi'}}


def _rows_in(A, B):
    """Boolean mask: which rows of ``A`` also occur in ``B``."""
    seen = {tuple(r) for r in B}
    return np.array([tuple(r) in seen for r in A], dtype=bool)


def _hamming(A, b):
    return np.mean(A != b, axis=1)


def _duplication_selection(index, pop, nd_pop, nxt_mask, ave):
    NDF, NDX = objs(nd_pop), decs(nd_pop)
    NXF, NXX = objs(pop[nxt_mask]), decs(pop[nxt_mask])
    F, X = objs(pop[index]), decs(pop[index])
    hit = np.where(NXF[:, 0] == F[0, 0])[0]
    if len(hit):
        d = _hamming(X, NXX[hit[0]])
    else:
        d = _hamming(X, NDX[int(np.argmin(np.abs(F[0, 0] - NDF[:, 0])))])
    ok = d >= ave
    return index[ok] if ok.any() else index[[int(np.argmax(d))]]


def _mop_selection(pop, N):
    """Bi-objective selection: unique decision vectors, first fronts first, duplicated objective vectors resolved by the
    decision-space distance to the selected / nearest first-front solutions, crowding on the last front."""
    X0 = np.round(decs(pop)[:N])
    D0 = np.mean(X0[:, None, :] != X0[None, :, :], axis=2)
    n0 = len(X0)
    ave = np.sum(np.tril(D0)) / max(1, n0 * (n0 - 1) // 2)
    pop = pop[np.unique(decs(pop), axis=0, return_index=True)[1]]
    front, _ = nd_sort(objs(pop), None, np.inf)
    nxt = front == 1
    nd_pop = pop[front == 1]
    top = int(front[np.isfinite(front)].max())
    while nxt.sum() < N or nxt.sum() == len(pop):
        before = int(nxt.sum())
        for i in range(2, top + 1):
            No = np.where(front == i)[0]
            No = No[~nxt[No]]
            F = objs(pop[No])
            if len(No) == 0:
                continue
            _, first, c = np.unique(F, axis=0, return_index=True, return_inverse=True)
            c = np.asarray(c).reshape(-1)
            order = np.argsort(np.argsort(first, kind="stable"), kind="stable")
            c = order[c]
            chosen = []
            for j in range(c.max() + 1):
                idx = np.where(c == j)[0]
                chosen.extend(_duplication_selection(No[idx], pop, nd_pop, nxt, ave) if len(idx) > 1 else No[idx])
            nxt[np.array(chosen, dtype=int)] = True
            if nxt.sum() >= N:
                break
        if int(nxt.sum()) == before:                 # nothing left to add (the reference would loop forever here)
            break
    pop, front = pop[nxt], front[nxt]
    maxf = front.max()
    keep = front < maxf
    cd = crowding(objs(pop), front)
    last = np.where(front == maxf)[0]
    keep[last[np.argsort(-cd[last], kind="stable")[: N - int(keep.sum())]]] = True
    return pop[keep], front[keep], cd[keep]


def _sop_selection(pop, N, alpha, z):
    F = objs(pop)
    fit = (F[:, 0] - z[0]) * (1 - alpha) + (F[:, 1] - z[1]) * alpha
    rank = np.argsort(fit, kind="stable")[:N]
    return pop[rank], fit[rank]


def _update_weight(pset, fmin, rng):
    zero = np.zeros(2)
    if np.array_equal(pset[0], pset[1]) and not np.array_equal(pset[0], zero):
        a1 = np.round(rng.random(), 2)
        with np.errstate(all="ignore"):
            a2 = np.round((pset[1, 0] - fmin[0]) / ((pset[1, 0] - fmin[0]) + (pset[0, 1] - fmin[1])), 2)
    elif np.array_equal(pset[0], zero) and np.array_equal(pset[1], zero):
        a1, a2 = np.round(rng.random(), 2), np.round(rng.random(), 2)
    else:
        with np.errstate(all="ignore"):
            a1 = np.round((pset[0, 0] - fmin[0]) / ((pset[0, 0] - fmin[0]) + (pset[0, 1] - fmin[1])), 2)
            a2 = np.round((pset[1, 0] - fmin[0]) / ((pset[1, 0] - fmin[0]) + (pset[1, 1] - fmin[1])), 2)
    return np.array([a1, a2])


def _partition_points(pop, front, pset):
    """Midpoints of the two largest angular gaps between consecutive first-front solutions."""
    F, ic = np.unique(objs(pop), axis=0, return_index=True)
    front = front[ic]
    fronts = np.unique(front[np.isfinite(front)])
    fr = np.where(front == fronts[0])[0]
    fmin = F[fr].min(axis=0)
    ang = np.zeros((len(F), len(F)))
    rank = np.lexsort((F[fr, 1], F[fr, 0]))
    for j in range(len(fr) - 1):
        a, b = F[fr[rank[j + 1]]] - fmin, F[fr[rank[j]]] - fmin
        with np.errstate(all="ignore"):
            cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        ang[fr[rank[j]], fr[rank[j + 1]]] = np.arccos(np.clip(cos, -1, 1)) if np.isfinite(cos) else np.nan

    def largest():
        m = np.nanmax(ang)
        r, c = np.where(ang == m)
        return r[0], c[0]

    if len(fr) == 1:
        part = np.zeros((2, 2))
    elif len(fr) == 2:
        r, c = largest()
        mid = (F[r] + F[c]) / 2
        part = np.vstack([mid, mid])
        ang[r, c] = 0
    else:
        r, c = largest()
        m1 = (F[r] + F[c]) / 2
        ang[r, c] = 0
        r, c = largest()
        m2 = (F[r] + F[c]) / 2
        ang[r, c] = 0
        part = np.vstack([m1, m2])
    for k in (0, 1):
        if _rows_in(part[[k]], pset)[0]:
            if np.nanmax(ang) != 0:
                r, c = largest()
                part[k] = (F[r] + F[c]) / 2
                ang[r, c] = 0
            else:
                part[k] = part[1 - k]
    return part


class MFFS(LoopAlgorithm):
    """Three cooperating populations: a Pareto (bi-objective) one and two single-objective sub-populations that follow
    scalarisations of (feature ratio, error) with weights re-derived from the widest gaps of the Pareto front; the
    weights and the sub-populations restart when their best values stop improving."""

    def _initialize_infill(self):
        rng, N, D = self.rng, self.N, self.D
        X = np.zeros((N, D))
        for i in range(N):
            k = int(rng.integers(1, int(round(D)) + 1))
            X[i, rng.permutation(D)[:k]] = 1
        return self.evaluate(X)

    def _initialize_advance(self, infills=None, **kwargs):
        N = self.N
        self.n0 = int(round(N * 0.5))
        self.n1 = int(round(N * 0.25))
        self.n2 = N - self.n0 - self.n1
        P = infills
        self.alpha = np.array([0.01, 0.99])
        self.pset = np.zeros((2, 2))
        self.pop, self.front, self.crowd = _mop_selection(P, self.n0)
        z = objs(P).min(axis=0)
        self.sub1, self.fit1 = _sop_selection(P, self.n1, self.alpha[0], z)
        self.sub2, self.fit2 = _sop_selection(P, self.n2, self.alpha[1], z)
        self.best = np.vstack([objs(self.sub1[[0]]), objs(self.sub2[[0]])])
        self._set_optimum()

    def _reproduce(self, parents, existing):
        if np.any(self.encoding != 4):
            raise ValueError("MFFS supports binary decision variables only")
        rng = self.rng
        P = decs(parents)
        h = len(P) // 2
        P1, P2 = P[:h], P[h: 2 * h]
        D = P.shape[1]
        k = rng.random((h, D)) < 0.5
        O1, O2 = P1.copy(), P2.copy()
        O1[k], O2[k] = P2[k], P1[k]
        O = np.vstack([O1, O2])
        site = rng.random((2 * h, D)) < 1.0 / D
        O[site] = 1 - O[site]
        zero = O.sum(axis=1) == 0
        if zero.any():
            O[zero] = rng.integers(0, 2, (int(zero.sum()), D))
        dup = _rows_in(O, decs(existing))
        for i in np.where(dup)[0]:
            on, off = np.where(O[i] > 0)[0], np.where(O[i] == 0)[0]
            if len(on):
                O[i, on[int(rng.integers(0, len(on)))]] = 0
            if len(off):
                O[i, off[int(rng.integers(0, len(off)))]] = 1
        O = np.vstack([O[~dup], O[dup]])
        O = np.unique(O, axis=0)
        O = O[O.sum(axis=1) > 0]
        return self.evaluate(O)

    def _bool_improvement(self, best_f, gen=5):
        flag = 0
        B = self.best
        if B.shape[1] / 2 < gen:
            B = np.hstack([B, best_f])
        else:
            for j in range(gen - 1):
                B[:, 2 * j: 2 * j + 2] = B[:, 2 * (j + 1): 2 * (j + 1) + 2]
            B[:, 2 * gen - 2: 2 * gen] = best_f
            if np.array_equal(B[:, :2], B[:, 2 * gen - 2: 2 * gen]):
                flag = 1
                B = np.zeros((2, 0))
        self.best = B
        return flag

    def _re_initialise(self, fmin):
        rng = self.rng
        X = decs(self.pop).copy()
        for i in range(len(X)):
            on, off = np.where(X[i] > 0)[0], np.where(X[i] == 0)[0]
            if len(on):
                X[i, on[int(rng.integers(0, len(on)))]] = 0
            if len(off):
                X[i, off[int(rng.integers(0, len(off)))]] = 1
        p = self.evaluate(X)
        self.sub1, self.fit1 = _sop_selection(p, self.n1, self.alpha[0], fmin)
        self.sub2, self.fit2 = _sop_selection(p, self.n2, self.alpha[1], fmin)

    def step(self):
        rng = self.rng
        pop, sub1, sub2 = self.pop, self.sub1, self.sub2
        mate = tournament(2, self.n0, self.front, -self.crowd, rng=rng)
        off = self._reproduce(pop[mate], Population.merge(pop, sub1, sub2))
        off1 = self._reproduce(sub1[tournament(2, self.n1, self.fit1, rng=rng)], Population.merge(pop, sub1, sub2, off))
        off2 = self._reproduce(sub2[tournament(2, self.n2, self.fit2, rng=rng)], Population.merge(pop, sub1, sub2, off, off1))
        self.pop, self.front, self.crowd = _mop_selection(Population.merge(pop, off, off1, off2), self.n0)
        fmin = objs(self.pop).min(axis=0)
        self.sub1, self.fit1 = _sop_selection(Population.merge(off, off2, sub1, off1), self.n1, self.alpha[0], fmin)
        self.sub2, self.fit2 = _sop_selection(Population.merge(off, off1, sub2, off2), self.n2, self.alpha[1], fmin)
        flag = self._bool_improvement(np.vstack([objs(self.sub1[[0]]), objs(self.sub2[[0]])]))
        if flag == 1:
            self.pset = _partition_points(self.pop, self.front, self.pset)
            self.alpha = _update_weight(self.pset, fmin, rng)
        elif not np.array_equal(self.alpha, [0.01, 0.99]):
            self.alpha = _update_weight(self.pset, fmin, rng)
        if flag == 1:
            self._re_initialise(fmin)
