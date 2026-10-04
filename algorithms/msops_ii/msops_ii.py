# emopylab 2026
"""MSOPS-II (multiple single objective Pareto sampling II).

Reference:
E. J. Hughes. MSOPS-II: A general-purpose many-objective optimiser. Proceedings of the IEEE Congress
on Evolutionary Computation, 2007, 3944-3951.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, objs
from core.population import Population

ALGORITHM_FLAGS = {'MSOPSII': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _metric(F, W, Zmin, Zmax):
    with np.errstate(all="ignore"):
        F = (F - Zmin) / (Zmax - Zmin)
        Wt = 1.0 / (W - Zmin + np.finfo(float).eps)
        Wn = Wt / np.linalg.norm(Wt, axis=1, keepdims=True)
        WMM = np.max(F[:, None, :] * Wn[None, :, :], axis=2)
        normP = np.linalg.norm(F, axis=1)
        VADS = normP[:, None] / ((F @ Wt.T) / normP[:, None]) ** 100
    return WMM, VADS


def cal_metric(F, W, want_vads=True):
    WMM, VADS = _metric(F, W, F.min(axis=0), F.max(axis=0))
    return (WMM, VADS) if want_vads else WMM


def _nan_cos(A):
    n = np.linalg.norm(A, axis=1)
    with np.errstate(all="ignore"):
        return (A @ A.T) / (n[:, None] * n[None, :])


def update_archive(archive, pop, popsize, rng):
    archive = Population.merge(archive, pop)
    F = objs(archive)
    N = len(archive)
    WMM = cal_metric(F, F, False)
    diag = np.diag(WMM).copy()
    remain = np.ones(N, bool)
    for i in range(N):
        if remain[i]:
            if WMM[i, i] > np.nanmin(WMM[remain, i]):
                remain[i] = False
            else:
                remain[WMM[i] < diag] = False
    archive = archive[remain]
    if len(archive) > 10 * popsize:
        archive = archive[rng.permutation(len(archive))[: 5 * popsize]]
    return archive


def update_weight(weight, F, K):
    W = np.vstack([weight, F]) if len(F) else weight
    N = len(W)
    Wn = W - W.min(axis=0)
    with np.errstate(all="ignore"):
        Wn = Wn / np.linalg.norm(Wn, axis=1, keepdims=True)
    cosine = _nan_cos(Wn)
    np.fill_diagonal(cosine, 0.0)
    idx = np.arange(N)
    while len(idx) > K:
        temp = np.sort(-cosine[np.ix_(idx, idx)], axis=1)
        idx = np.delete(idx, np.lexsort(temp.T[::-1])[0])
    return W[idx]


def environmental_selection(pop, weight, K):
    F, C = objs(pop), cons(pop)
    WMM, VADS = cal_metric(F, weight)
    S = np.hstack([WMM, VADS])
    minidx = np.argmin(np.where(np.isnan(S), np.inf, S), axis=0)
    minval = S[minidx, np.arange(S.shape[1])]
    with np.errstate(all="ignore"):
        S1 = S / minval
        for i, mi in enumerate(minidx):
            others = np.delete(S[:, i], mi)
            S1[mi, i] = S[mi, i] / np.nanmin(others)
        r = np.nanmin(S1, axis=1)
    feasible = np.where(np.all(C <= 0, axis=1))[0] if C.size else np.arange(len(pop))
    if len(feasible):
        ext = feasible[np.argmin(F[feasible], axis=0)]
        r[ext] = r[ext] - (np.nanmax(r) - np.nanmin(r))
    if C.size:
        infe = np.where(np.any(C > 0, axis=1))[0]
        r[infe] = np.sum(C[infe] ** 2, axis=1) + np.nanmax(r)
    return pop[np.argsort(np.where(np.isnan(r), np.inf, r), kind="stable")[:K]]


def _mating_selection(pop, archive, rng):
    N, NA = len(pop), len(archive)
    p1, p2 = rng.integers(0, NA, N), rng.integers(0, NA, N)
    temp = rng.random(N) < 0.5
    choose1 = np.ones(N, bool)
    Fp, Xp, Fa, Xa = objs(pop), decs(pop), objs(archive), decs(archive)
    choose1[temp] = np.sum((Fa[p1[temp]] - Fp[temp]) ** 2, axis=1) < np.sum((Fa[p2[temp]] - Fp[temp]) ** 2, axis=1)
    nt = ~temp
    choose1[nt] = np.sum((Xa[p1[nt]] - Xp[nt]) ** 2, axis=1) < np.sum((Xa[p2[nt]] - Xp[nt]) ** 2, axis=1)
    pool = np.concatenate([p1[choose1], p2[~choose1]])
    return Population.merge(pop, archive[pool])


class MSOPSII(LoopAlgorithm):
    """Multiple single-objective Pareto sampling (second version): solutions are ranked by their best score over
    a set of adaptively chosen targets (weighted min-max and vector-angle metrics), with an external archive."""

    def start(self):
        pop, N = self.pop, self.N
        C = cons(pop)
        feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
        if feas.any():
            self.archive = update_archive(None, pop[feas], N, self.rng)
            self.weight = update_weight(np.zeros((0, self.M)), objs(pop[feas]), N)
        else:
            best = int(np.argmin(np.sum(np.maximum(0, C), axis=1)))
            self.archive = pop[[best]]
            self.weight = objs(pop[[best]])

    def step(self):
        N = self.N
        parents = _mating_selection(self.pop, self.archive, self.rng)
        off = self.evaluate(ga_half(self.problem, decs(parents), rng=self.rng))
        C = cons(off)
        feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(off), bool)
        self.archive = update_archive(self.archive, off[feas], N, self.rng)
        self.weight = update_weight(self.weight, objs(off[feas]), N)
        self.pop = environmental_selection(Population.merge(self.pop, off), self.weight, N)
