# emopylab 2026
"""SAMOEA-TL2M (two-level model management based surrogate-assisted MOEA).

Reference:
Y. Liu, J. Ding, Q. Li, F. Li, and J. Liu. A two-level model management-based surrogate-assisted evolutionary
algorithm for medium-scale expensive multiobjective optimization. IEEE Transactions on Systems, Man, and Cybernetics:
Systems, 2025.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs
from algorithms.hee_moea.hee_moea import ClusterRBF
from core.population import Population
from operators.utility_functions.UniformPoint import UniformPoint

ALGORITHM_FLAGS = {"SAMOEA_TL2M": {"multi", "many"}}


def _pdist(A, B):
    return np.sqrt(np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0))


def _minmax(F):
    with np.errstate(all="ignore"):
        return (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))


def sde_score(F, chunk=512):
    """Shift-based density of every row (objectives normalised to [0,1] per column), rescaled to [0,1]:
    min over j != k of ||f_k - max(f_j, f_k)||."""
    P = _minmax(np.asarray(F, dtype=float))
    n = len(P)
    sde = np.empty(n)
    for s in range(0, n, chunk):
        k = np.arange(s, min(n, s + chunk))
        d = np.linalg.norm(P[k, None, :] - np.maximum(P[None, :, :], P[k, None, :]), axis=2)
        d[np.arange(len(k)), k] = np.inf
        sde[k] = d.min(axis=1)
    with np.errstate(all="ignore"):
        return (sde - sde.min()) / (sde.max() - sde.min())


def idw_uncertainty(x, known, values, p=2):
    """Inverse-distance-weighted variance of ``values`` around its IDW prediction at every row of ``x``."""
    d = _pdist(np.atleast_2d(x), known)
    with np.errstate(all="ignore"):
        w = 1.0 / d ** p
        w = w / w.sum(axis=1, keepdims=True)
        pred = w @ values
        return np.sum(w * (values[None, :] - pred[:, None]) ** 2, axis=1)


def _pick_spread(dec, order, n):
    """First ``n`` rows of ``dec[order]`` skipping those within 1e-5 of an already picked one (stops if exhausted)."""
    out = [dec[order[0]]]
    h = 1
    while len(out) < n and h < len(order):
        cand = dec[order[h]]
        if _pdist(cand[None], np.asarray(out)).min() > 1e-5:
            out.append(cand)
        h += 1
    return np.asarray(out)


class SAMOEA_TL2M(LoopAlgorithm):
    """RBF surrogates of the normalised objectives and of the shift-based density drive a 20-generation surrogate
    search from the NP-member population A1; KE new points are chosen from the predicted first front by density
    (level 1) or, in the uncertainty stage, partly by the IDW variance (level 2); the stage switches when the accuracy
    rate indicator (share of infill points kept in A1 and non-dominated there) falls below ``alpha``."""

    OBJECTIVE_SCOPE = "many"

    def __init__(self, pop_size: int = 100, G: int = 20, KE: int = 5, alpha: float = 0.4, NP: int = 100,
                 sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.G, self.KE, self.alpha, self.NP = int(G), int(KE), float(alpha), int(NP)

    def _initialize_infill(self):
        self.pop_size = UniformPoint(self.pop_size, self.M)[1]
        NI = 11 * self.D - 1
        P, _ = UniformPoint(NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.A2 = infills
        self.A1 = infills[self.rng.integers(0, len(infills), self.NP)]         # with replacement
        self.stage = 1
        self.pop = self.A2
        self._set_optimum()

    def _surrogate_search(self):
        M, D = self.M, self.D
        k = int(round(np.sqrt(M + D) + 3))
        X, Y = decs(self.A2), objs(self.A2)
        with np.errstate(all="ignore"):
            yy = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
            PopObj = (objs(self.A1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
        models = [ClusterRBF(k, self.rng).fit(X, yy[:, [i]]) for i in range(M)]
        model_c = ClusterRBF(k, self.rng).fit(X, sde_score(Y)[:, None])
        PopDec = decs(self.A1)
        for _ in range(self.G):
            Off = ga(self.problem, PopDec, rng=self.rng)
            PopDec = np.vstack([PopDec, Off])
            PopObj = np.vstack([PopObj, np.column_stack([m.predict(Off).ravel() for m in models])])
            front, max_f = nd_sort(PopObj, None, len(Off))
            nxt = front < max_f
            pd = model_c.predict(PopDec).ravel()
            last = np.where(front == max_f)[0]
            rank = np.argsort(-pd[last], kind="stable")
            nxt[last[rank[: len(Off) - int(nxt.sum())]]] = True
            PopObj, PopDec = PopObj[nxt], PopDec[nxt]
        return PopDec, PopObj

    def step(self):
        KE, NP = self.KE, self.NP
        PopDec, PopObj = self._surrogate_search()
        keep = _pdist(PopDec, decs(self.A2)).min(axis=1) >= 1e-5
        PopDec1, PopObj = PopDec[keep], PopObj[keep]
        if len(PopDec1) == 0:
            P, _ = UniformPoint(1, self.D, "Latin", rng=self.rng)             # literal: not scaled to the bounds
            self.A2 = Population.merge(self.A2, self.evaluate(np.asarray(P)))
            self.pop = self.A2
            return
        infill_c = np.zeros((0, self.D))
        if self.stage == 2:
            NumC = max(1, int(np.floor(KE * (1 - self.alpha) + 0.5)))
            last = np.where(nd_sort(PopObj, None, 1)[0] == 1)[0]
            A2d, A2o = decs(self.A2), objs(self.A2)
            s = np.sum([idw_uncertainty(PopDec1, A2d, A2o[:, h]) for h in range(self.M)], axis=0)
            rank = np.argsort(-s[last], kind="stable")
            infill_c = _pick_spread(PopDec1, last[rank], NumC) if len(PopDec1) > NumC else PopDec1
        Num = KE if self.stage == 1 else KE - NumC
        front, max_f = nd_sort(PopObj, None, 1)
        last = np.where(front == max_f)[0]
        if len(last) <= Num:
            infill = PopDec1[last]
        else:
            pd = sde_score(np.vstack([objs(self.A2), PopObj]))[len(self.A2):]   # literal: raw archive + normalised
            rank = np.argsort(-pd[last], kind="stable")
            infill = _pick_spread(PopDec1, last[rank], Num) if len(PopDec1) > Num else PopDec1
        if self.stage == 2:
            infill = np.vstack([infill, infill_c])
        if len(infill) == 0:
            self.stage = 2
            self.pop = self.A2
            return
        New = self.evaluate(infill)
        self.A2 = Population.merge(self.A2, New)
        nextA1 = Population.merge(self.A1, New)
        F, C = objs(nextA1), cons(nextA1)
        front, max_f = nd_sort(F, C if C.shape[1] else None, NP)
        nxt = front < max_f
        cd = sde_score(F)
        last = np.where(front == max_f)[0]
        rank = np.argsort(-cd[last], kind="stable")
        nxt[last[rank[: NP - int(nxt.sum())]]] = True
        self.A1 = nextA1[nxt]
        n_new = int(nxt[NP:].sum())
        rp = n_new / len(infill)
        fA1 = nd_sort(objs(self.A1), C[nxt] if C.shape[1] else None, NP)[0]
        with np.errstate(all="ignore"):
            rn = np.sum(fA1[NP - n_new:] == 1) / n_new if n_new else np.nan
        ari = rp if np.isnan(rn) else min(rp, rn)                      # MATLAB min ignores NaN
        self.stage = 2 if ari < self.alpha else 1
        self.pop = self.A2
