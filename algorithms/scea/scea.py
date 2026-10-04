# emopylab 2026
"""SCEA (sparsity clustering basec evolutionary algorithm).

Reference:
Y. Zhang, C. Wu, Y. Tian, and X. Zhang. A co-evolutionary algorithm based on sparsity clustering for
sparse large-scale multi-objective optimization. Engineering Applications of Artificial
Intelligence, 2024, 133: 108194
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, ga_half, kmeans, nd_sort, objs, tournament
from algorithms.mskea.mskea import _spea2_selection
from core.population import Population

ALGORITHM_FLAGS = {'SCEA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


def theta_mid(theta_all, rng):
    """Median of the k-means centroids of the mask sizes (k close to the number of non-dominated solutions)."""
    n = len(theta_all)
    if n == 0:
        return 0.0
    k = max(1, n - 1) if len(np.unique(theta_all)) % 2 == 0 else max(1, n - 2)
    k = min(k, n)
    lab = kmeans(theta_all.reshape(-1, 1).astype(float), k, rng)
    cent = np.array([theta_all[lab == c].mean() for c in np.unique(lab)])
    return float(np.median(cent))


def init_rank(pop, Dec, Mask):
    """Order a sub-population by non-domination level and, inside a level, by decreasing crowding distance."""
    n = len(pop)
    if n == 0:
        return pop, Dec, Mask, np.arange(0)
    C = cons(pop)
    front, _ = nd_sort(objs(pop), C if C.size else None, n)
    cd = crowding(objs(pop), front)
    order = np.lexsort((-cd, front))
    return pop[order], Dec[order], Mask[order], np.arange(1, n + 1)


def _ts(rng, f, k):
    return np.zeros(0, dtype=int) if len(f) == 0 or k <= 0 else tournament(2, int(k), f, rng=rng)


def operator_win(algo, dec, mask, rank):
    rng, D = algo.rng, algo.D
    pool = tournament(2, 2 * len(rank), rank, rng=rng)
    pd, pm = dec[pool], mask[pool]
    n = len(pd)
    perm = rng.permutation(n)
    P1m, P2m = pm[perm[: n // 2]], pm[perm[n // 2:]]
    k = rng.random((n // 2, D)) < 0.5
    rng.random(n // 2)
    off = P1m.copy()
    off[k] = P2m[k]
    site = rng.random((n // 2, D)) < 1.0 / D
    off[site] = ~off[site].astype(bool)
    return ga_half(algo.problem, pd, rng=rng), off


def _operator_pair(algo, windec, winmask, winrank, sdec, smask, srank, fit, thetamid, upper_side):
    """``upper_side`` False: masks below the threshold grow (Min); True: masks above it shrink (Max)."""
    rng, D = algo.rng, algo.D
    n = len(srank)
    wp = tournament(2, n, winrank, rng=rng)
    mp = tournament(2, n, srank, rng=rng)
    pdec = np.vstack([windec[wp], sdec[mp]])
    P1m, P2m = winmask[wp], smask[mp]
    k = rng.random((n, D)) < 0.5
    rng.random(n)
    off = P1m.copy()
    off[k] = P2m[k]
    for i in range(n):
        active = int(off[i].sum())
        if (active <= thetamid) if upper_side else (active >= thetamid):
            site = rng.random(D) < 1.0 / D
            off[i, site] = ~off[i, site].astype(bool)
        elif upper_side:
            idx1 = np.where(off[i] != 0)[0]
            sel = idx1[_ts(rng, -fit[idx1], int(np.floor(len(idx1) - thetamid)))]
            off[i, sel] = 0
        else:
            idx0 = np.where(off[i] == 0)[0]
            sel = idx0[_ts(rng, fit[idx0], int(np.floor(thetamid - active)))]
            off[i, sel] = 1
    return ga_half(algo.problem, pdec, rng=rng), off


def operator_min(algo, *a):
    return _operator_pair(algo, *a, upper_side=False)


def operator_max(algo, *a):
    return _operator_pair(algo, *a, upper_side=True)


def initial_population(algo, mask_size_fn):
    """Variable probing, the nested-mask population and the sparse random population (SGECF / SCEA share it)."""
    N, D, rng = algo.N, algo.D, algo.rng
    lo, up = algo.lower, algo.upper
    TDec, TMask, TPop = [], [], []
    Fit = np.zeros(D)
    for _ in range(4):
        Dec = lo + rng.random((D, D)) * (up - lo)
        Mask = np.eye(D)
        P = algo.evaluate(Dec * Mask)
        TDec.append(Dec), TMask.append(Mask), TPop.append(P)
        C = cons(P)
        Fit = Fit + nd_sort(np.hstack([objs(P), C]) if C.size else objs(P), None, np.inf)[0]
    VDec = lo + rng.random((D, D)) * (up - lo)
    VMask = np.zeros((D, D))
    order = np.argsort(Fit, kind="stable")
    for i in range(D):
        VMask[i, order[: i + 1]] = 1
    VPop = algo.evaluate(VDec * VMask)
    VPop, VDec, VMask, front, _ = _spea2_selection(Population.merge(VPop, *TPop), np.vstack([VDec] + TDec), np.vstack([VMask] + TMask), N)
    K = mask_size_fn(VMask, front, Fit)
    Dec = lo + rng.random((N, D)) * (up - lo)
    Mask = np.zeros((N, D))
    for i in range(N):
        Mask[i, tournament(2, K(), Fit, rng=rng)] = 1
    P = algo.evaluate(Dec * Mask)
    pop, algo.Dec, algo.Mask, algo.front, algo.crowd = _spea2_selection(
        Population.merge(VPop, P), np.vstack([VDec, Dec]), np.vstack([VMask, Mask]), N)
    algo.fitness = Fit
    return pop


class SCEA(LoopAlgorithm):
    """Sparse-scale co-evolution: the non-dominated masks define a target sparsity level (the median of clustered
    mask sizes); the dominated solutions above/below it are pulled toward it while the non-dominated ones recombine."""

    def _initialize_infill(self):
        rng = self.rng

        def mask_size(VMask, front, Fit):
            tm = theta_mid(VMask[front == 1].sum(axis=1), rng)
            return lambda: int(np.round(tm))
        return initial_population(self, mask_size)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _offspring(self, sub, tm):
        (pw, dw, mw, rw), (p1, d1, m1, r1), (p2, d2, m2, r2) = sub
        n1, n2 = len(p1), len(p2)
        N = self.N
        F = self.fitness
        if n1 > n2 and n1 > N // 3:
            o1 = operator_win(self, np.vstack([dw, d2]), np.vstack([mw, m2]), np.concatenate([rw, r2 + (rw.max() if len(rw) else 0)]))
            o2 = operator_min(self, dw, mw, rw, d1, m1, r1, F, tm)
            o3 = None
        elif n1 < n2 and n2 > N // 3:
            o1 = operator_win(self, np.vstack([dw, d1]), np.vstack([mw, m1]), np.concatenate([rw, r1 + (rw.max() if len(rw) else 0)]))
            o3 = operator_max(self, dw, mw, rw, d2, m2, r2, F, tm)
            o2 = None
        else:
            o1 = operator_win(self, dw, mw, rw)
            o2 = operator_min(self, dw, mw, rw, d1, m1, r1, F, tm) if n1 > 0 else None
            o3 = operator_max(self, dw, mw, rw, d2, m2, r2, F, tm) if n2 > 0 else None
        parts = [o for o in (o1, o2, o3) if o is not None]
        return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])

    def _split(self):
        win = np.where(self.front == 1)[0]
        tm = theta_mid(self.Mask[win].sum(axis=1), self.rng)
        lose = np.where(self.front != 1)[0]
        lt = self.Mask[lose].sum(axis=1)
        groups = [win, lose[lt <= tm], lose[lt > tm]]
        sub = []
        for g in groups:
            p, d, m, r = init_rank(self.pop[g], self.Dec[g], self.Mask[g])
            sub.append((p, d, m, r))
        return sub, tm

    def step(self):
        sub, tm = self._split()
        OffDec, OffMask = self._offspring(sub, tm)
        off = self.evaluate(OffDec * OffMask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd = _spea2_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), self.N)
