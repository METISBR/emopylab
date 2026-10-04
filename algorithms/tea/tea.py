# emopylab 2026
"""TEA (two-phase evolutionary algorithm).

Reference:
Z. Zhang, Y. Wang, J. Liu, G. Sun, and K. Tang. A two-phase Kriging- assisted evolutionary algorithm
for expensive constrained multiobjective optimization problems. IEEE Transactions on Systems, Man,
and Cybernetics: Systems, 2024, 54(8): 4579-4591.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, truncate_lexi
from algorithms.community_utils.dace import DaceModel, norm_cdf
from algorithms.community_utils.pea import _dist, _nanargmax
from core.population import Population

ALGORITHM_FLAGS = {'TEA': {'constrained', 'expensive', 'integer', 'multi', 'real'}}


def nd_sort_pdpd(F, FM, n_sort, C=None, CM=None, eps=0.75):
    """Sorting by probabilistic dominance: objectives whose pairwise win probabilities differ by at most ``eps`` are
    aggregated into one product term; with constraint models, pairs not both likely feasible are ordered by the total
    probability of feasibility."""
    N = len(F)
    i_idx, j_idx = np.triu_indices(N, 1)
    with np.errstate(all="ignore"):
        Pi = norm_cdf(-(F[i_idx] - F[j_idx]) / np.sqrt(FM[i_idx] + FM[j_idx]))
    Pj = 1 - Pi
    agg = np.abs(Pi - Pj) <= eps
    PDi, PDj = np.where(agg, Pi, 1.0).prod(1), np.where(agg, Pj, 1.0).prod(1)
    a, b = -Pi, -Pj
    le = np.all((a <= b) | agg, 1) & (-PDi <= -PDj)
    ge = np.all((a >= b) | agg, 1) & (-PDi >= -PDj)
    eq = np.all((a == b) | agg, 1) & (-PDi == -PDj)
    flag = np.where(le & ~eq, 1, np.where(ge & ~eq, 2, 3))
    if C is not None:
        with np.errstate(all="ignore"):
            pf = norm_cdf(-C / np.sqrt(CM))
        lpof = np.ones(N)
        for j in range(C.shape[1]):
            lpof = np.fmin(lpof, pf[:, j])
        tpof = pf.prod(1) if C.shape[1] else np.ones(N)
        both = ((lpof[i_idx] >= 0.5) & (lpof[j_idx] >= 0.5)) | (lpof[i_idx] == lpof[j_idx])
        ti, tj = tpof[i_idx], tpof[j_idx]
        by_t = np.where(np.isnan(ti) & ~np.isnan(tj), 2, np.where(~np.isnan(tj) & (tj > ti), 2, 1))
        flag = np.where(both, flag, by_t)
    dom = np.zeros((N, N), bool)
    dom[i_idx[flag == 1], j_idx[flag == 1]] = True
    dom[j_idx[flag == 2], i_idx[flag == 2]] = True
    front, maxf = np.full(N, np.inf), 0
    while np.sum(front != np.inf) < min(n_sort, N):
        maxf += 1
        cur = np.where(front == np.inf)[0]
        cnt = dom[np.ix_(cur, cur)].sum(0)
        idx = cur[cnt == cnt.min()]
        front[idx] = maxf
        dom[idx, :] = False
    return front, maxf


def _norm(F, *extra):
    lo, hi = F.min(0), F.max(0)
    r = np.maximum(hi - lo, 10e-10)
    return (F - lo) / r, r


def _dis_selection(P, last, mu, zero_diag):
    Dm = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    np.fill_diagonal(Dm, 0.0 if zero_diag else np.inf)
    D = 1.0 / (np.sort(Dm, 1)[:, 0] + 2)
    return np.argsort(D[last], kind="stable")[:mu]


def _survive(F, front, maxf, N, zero_diag):
    nxt = front < maxf
    last = np.where(front == maxf)[0]
    if maxf == 1:
        nxt[last[~truncate_lexi(_dist(F[last], False), len(last) - N)]] = True
    else:
        nxt[last[_dis_selection(F, last, N - int(nxt.sum()), zero_diag)]] = True
    return nxt


def _set_dominate(A, B):
    fb, _ = nd_sort(B, None, np.inf)
    fa, _ = nd_sort(A, None, np.inf)
    A, B = A[fa == 1], B[fb == 1]
    rows = []
    for a in A:
        rel = np.where(np.all(a == B, 1), 3, np.where(np.all(a <= B, 1), 1, np.where(np.all(a >= B, 1), 2, 3)))
        u = sorted(set(rel.tolist()))
        rows.append(u[0] if len(u) == 1 else (1 if u == [1, 3] else 2 if u == [2, 3] else 4))
    u = sorted(set(rows))
    return u[0] if len(u) == 1 else (1 if u == [1, 3] else 2 if u == [2, 3] else 4)


class TEA(LoopAlgorithm):
    """Kriging models of objectives (and, in the second phase, of constraints) drive a surrogate GA ranked by probabilistic
    dominance; up to ``mu`` candidates far from the evaluated set are sampled per iteration. The search switches from the
    unconstrained to the constrained phase once new samples have stopped improving the feasible front ``ct_max`` times."""

    def __init__(self, pop_size: int = 100, wmax: int = 20, mu: int = 5, ct_max: int = 2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax, self.mu, self.ct_max = int(wmax), int(mu), int(ct_max)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.NI = self.N
        P, _ = UniformPoint(self.NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop, self.P = infills, infills
        nc = cons(infills).shape[1]
        self.th_obj, self.th_con = 5.0 * np.ones((self.M, self.D)), 5.0 * np.ones((nc, self.D))
        self.phase, self.ct, self.success = 1, 0, True
        self.mo, self.mc = [], []
        self._set_optimum()

    def _fit(self, X, y, th):
        D = self.D
        d1 = np.unique(np.round(X * 1e12) / 1e12, axis=0, return_index=True)[1]
        d2 = np.unique(np.round(y * 1e12) / 1e12, return_index=True)[1]
        dist = np.intersect1d(d1, d2)
        if len(dist) <= D + 1:
            dist = d1      # deviation (see community_utils/pea.py): keep the fit determined for degenerate responses
        m = DaceModel(X[dist], y[dist], "regpoly1", th, 1e-5 * np.ones(D), 100 * np.ones(D))
        return m, m.theta

    def _train(self):
        X, F, C = decs(self.pop), objs(self.pop), cons(self.pop)
        self.mo = []
        for i in range(F.shape[1]):
            m, self.th_obj[i] = self._fit(X, F[:, i], self.th_obj[i])
            self.mo.append(m)
        self.mc = []
        if self.phase == 2:
            for i in range(C.shape[1]):
                m, self.th_con[i] = self._fit(X, C[:, i], self.th_con[i])
                self.mc.append(m)

    def _predict(self, X):
        po = [m.predict(X, mse=True) for m in self.mo]
        F, FM = np.column_stack([p[0] for p in po]), np.column_stack([p[1] for p in po])
        nc = cons(self.pop).shape[1]
        if self.phase == 2 and self.mc:
            pc = [m.predict(X, mse=True) for m in self.mc]
            return F, FM, np.column_stack([p[0] for p in pc]), np.column_stack([p[1] for p in pc])
        return F, FM, np.zeros((len(X), nc)), np.zeros((len(X), nc))

    def _pdpd(self, F, FM, C, CM, n):
        return nd_sort_pdpd(F, FM, n, C, CM) if self.phase == 2 else nd_sort_pdpd(F, FM, n)

    def _evo_search(self):
        X, F, C = decs(self.P), objs(self.P), cons(self.P)
        FM, CM = np.zeros_like(F), np.zeros_like(C)
        n = len(self.P)
        for _ in range(self.wmax):
            od = ga(self.problem, X, rng=self.rng)
            of, ofm, oc, ocm = self._predict(od)
            X, F, C = np.vstack([X, od]), np.vstack([F, of]), np.vstack([C, oc])
            FM, CM = np.vstack([FM, ofm]), np.vstack([CM, ocm])
            Fn, r = _norm(F)
            front, maxf = self._pdpd(Fn, FM / r ** 2, C, CM, n)
            idx = np.where(_survive(Fn, front, maxf, n, zero_diag=True))[0]
            X, F, C, FM, CM = X[idx], F[idx], C[idx], FM[idx], CM[idx]
        return X, F, C, FM, CM

    def _select(self, F, FM, C, CM):
        mu, NI = self.mu, self.NI
        AF, AC = objs(self.pop), cons(self.pop)
        allf = np.vstack([AF, F])
        lo, r = allf.min(0), np.maximum(allf.max(0) - allf.min(0), 10e-10)
        AF, F, FM = (AF - lo) / r, (F - lo) / r, FM / r ** 2
        if self.phase == 2:
            num = int(np.all(AC <= 0, 1).sum())
            fr, _ = nd_sort(AF, AC if AC.shape[1] else None, np.inf)
            if num > NI:
                A2 = AF[fr == 1]
            else:
                sel, i = fr == 1, 1
                top = np.nanmax(np.where(np.isinf(fr), np.nan, fr))
                while sel.sum() <= NI and i <= top:
                    sel |= fr == i
                    i += 1
                A2 = AF[sel]
        else:
            fr, _ = nd_sort(AF, None, np.inf)
            A2 = AF[fr == 1]
        front, maxf = self._pdpd(F, FM, C, CM, mu)
        nxt = front < maxf
        last = list(np.where(front == maxf)[0])
        if len(last) == mu - nxt.sum():
            nxt[last] = True
        elif len(last) > mu - nxt.sum():
            A2 = np.vstack([A2, F[nxt]])
            for _ in range(mu - int(nxt.sum())):
                d = np.sqrt(((F[last][:, None] - A2[None]) ** 2).sum(-1)).min(1)
                k = _nanargmax(d)
                nxt[last[k]] = True
                A2 = np.vstack([A2, F[last[k]]])
                del last[k]
        return nxt

    def _phase_trans(self, A2, Cp):
        if self.phase != 1:
            return
        feas = np.all(cons(A2) <= 0, 1)
        index = 0
        if feas.any():
            cf = np.all(cons(Cp) <= 0, 1)
            FA = objs(A2)[feas]
            FC = objs(Cp)
            ok_f = (not cf.any()) or _set_dominate(FC[cf], FA) == 3
            ok_i = (cf.all()) or _set_dominate(FC[~cf], FA) in (1, 3)
            if ok_f and ok_i:
                self.ct += 1
                if self.ct >= self.ct_max:
                    index = 1
            else:
                self.ct = 0
        self.phase = 2 if (feas.any() and index == 1) else 1

    def step(self):
        if self.success:
            self._train()
        X, F, C, FM, CM = self._evo_search()
        db = decs(self.pop)
        inside = np.array([np.any(np.all(db == x, 1)) for x in X])
        cand = np.zeros((0, self.D))
        if not inside.all():
            idx = np.where(~inside)[0]
            cand = X[idx[self._select(F[idx], FM[idx], C[idx], CM[idx])]]
        cand = np.array([c for c in cand if np.sqrt(((db - c) ** 2).sum(1)).min() > 1e-5]).reshape(-1, self.D)
        self.success = False
        if len(cand):
            new = self.evaluate(cand)
            self.success = True
            self._phase_trans(self.pop, new)
            self.pop = Population.merge(self.pop, new)
        F, C = objs(self.pop), cons(self.pop)
        Fn, _ = _norm(F)
        use_c = self.phase == 2 and C.shape[1]
        front, maxf = nd_sort(Fn, C if use_c else None, self.NI)
        self.P = self.pop[_survive(Fn, front, maxf, self.NI, zero_diag=False)]
