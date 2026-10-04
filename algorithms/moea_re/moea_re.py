# emopylab 2026
"""MOEA-RE (multi-objective evolutionary algorithm with robustness enhancement).

Reference:
Z. He, G. G. Yen, and J. Lv. Evolutionary multiobjective optimization with robustness enhancement.
IEEE Transactions on Evolutionary Computation, 2020, 24(3): 494-507.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga, nd_sort, objs, tournament, uniform_point
from algorithms.community_utils.robust import cosine_dist, perturbed_objs
from core.population import Population

ALGORITHM_FLAGS = {'MOEARE': {'binary', 'integer', 'label', 'multi', 'permutation', 'real', 'robust'}}


def _nan_argmin(A):
    """Row-wise argmin ignoring NaN (index 0 for an all-NaN row, as the reference's ``min`` does)."""
    A = np.where(np.isnan(A), np.inf, A)
    return np.argmin(A, axis=1)


def _mode(v):
    if len(v) == 0:
        return np.nan
    vals, cnt = np.unique(v, return_counts=True)
    return vals[np.argmax(cnt)]


def _select(pop, N, F):
    front, maxf = nd_sort(F, None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt], F[nxt]


def _select_soi(pop, F, N, z):
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.arccos(np.clip(1.0 - cosine_dist(F, F), -1.0, 1.0))
        np.fill_diagonal(angle, np.inf)
        asf = np.max((F - z) * F.sum(axis=1, keepdims=True) / F, axis=1)
    remain = list(range(len(F)))
    while len(remain) > N:
        sub = angle[np.ix_(remain, remain)]
        x = np.argmin(sub, axis=1)
        dis = sub[np.arange(len(remain)), x]
        y = int(np.argmin(dis))
        x = int(x[y])
        if asf[remain[x]] > asf[remain[y]]:
            del remain[x]
        else:
            del remain[y]
    remain = np.array(remain)
    return pop[remain], F[remain]


class MOEARE(LoopAlgorithm):
    """Robust search keeping (i) a population ranked on disturbed objectives and (ii) an archive of solutions of
    interest tied to reference vectors; the final answer picks, per reference vector, the archived solution with the
    best history of (disturbed) convergence."""

    def __init__(self, pop_size: int = 100, alpha: float = 1.5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha = float(alpha)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        _, self.front, self.crowd, _ = _select(self.pop, self.N, F)
        self.z = F.min(axis=0)
        self.arc = self.pop
        self.arc_w = [[int(k)] for k in _nan_argmin(cosine_dist(objs(self.arc) - self.z, self.W))]
        self.arc_sp = [[float(v)] for v in objs(self.arc).sum(axis=1)]

    def _update_archive(self, arc_F, soi, soi_F):
        z, W = self.z, self.W
        tr = np.max(np.sum(np.abs(soi_F - z), axis=1))
        remain = np.sum(np.abs(arc_F - z), axis=1) <= self.alpha * tr
        soi_w = _nan_argmin(cosine_dist(soi_F - z, W))
        arc_w_new = _nan_argmin(cosine_dist(arc_F - z, W))
        arc_sp_new = arc_F.sum(axis=1)
        self.arc_w = [w + [int(arc_w_new[i])] for i, w in enumerate(self.arc_w)]
        self.arc_sp = [s + [float(arc_sp_new[i])] for i, s in enumerate(self.arc_sp)]
        keep = np.where(remain)[0]
        self.arc = Population.merge(self.arc[keep], soi) if len(keep) else soi
        self.arc_w = [self.arc_w[i] for i in keep] + [[int(k)] for k in soi_w]
        self.arc_sp = [self.arc_sp[i] for i in keep] + [[float(v)] for v in soi_F.sum(axis=1)]

    def _final_selection(self):
        ArcW, ArcSP = [list(w) for w in self.arc_w], self.arc_sp
        K = len(self.W)
        genx = np.array([len(w) for w in ArcW], dtype=float)
        a_sp = np.array([np.sum(np.array(s) - np.min(s)) for s in ArcSP])
        SM = a_sp.mean()
        GM = min(30.0, genx.mean())
        S = np.array([np.mean(s) for s in ArcSP]) + SM / GM * np.maximum(0.0, GM - genx)
        selected = np.full(K, -1)
        while True:
            closest = np.array([_mode(w) for w in ArcW], dtype=float)
            assigned = []
            for i in np.where(selected < 0)[0]:
                cur = np.where(closest == i)[0]
                if len(cur):
                    selected[i] = cur[int(np.argmin(S[cur]))]
                    assigned.append(int(i))
            ArcW = [[v for v in w if v not in assigned] for w in ArcW]
            if not assigned:
                break
        return self.arc[selected[selected >= 0]]

    def step(self):
        rng, N = self.rng, self.N
        pop = self.pop
        X = decs(pop)
        off = self.evaluate(ga(self.problem, X[tournament(2, N, self.front, -self.crowd, rng=rng)], rng=rng))
        self.z = np.minimum(self.z, objs(off).min(axis=0))
        n1, n2 = len(pop), len(off)
        if hasattr(self.problem, 'perturb'):
            allF = perturbed_objs(self.problem, np.vstack([X, decs(off), decs(self.arc)]), 1, rng=rng)[0]
        else:
            allF = objs(self.evaluate(np.vstack([X, decs(off), decs(self.arc)])))
        popF, offF, arcF = allF[:n1], allF[n1: n1 + n2], allF[n1 + n2:]
        pop, self.front, self.crowd, popF = _select(Population.merge(pop, off), N, np.vstack([popF, offF]))
        soi, soi_F = _select_soi(pop, popF, 20, self.z)
        self._update_archive(arcF, soi, soi_F)
        if self.FE >= self.max_FE:
            pop = self._final_selection()
        self.pop = pop
