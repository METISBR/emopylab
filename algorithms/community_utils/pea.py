"""Kriging-assisted constrained EA with probabilistic dominance (PEA / PEA+): models for every objective and constraint,
surrogate evolution ranked by probabilistic dominance weighted by the probability of feasibility, and distance-based infill."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, truncate_lexi
from algorithms.community_utils.dace import DaceModel, norm_cdf
from core.population import Population

__all__ = ["PEABase", "nd_sort_cppd"]


def _nanargmax(v):
    return 0 if np.all(np.isnan(v)) else int(np.nanargmax(v))


def _dist(P, shifted):
    """Pairwise distance matrix (shift-based: d(i, j) = ||P_i - max(P_j, P_i)|| when ``shifted``); diagonal = inf."""
    if shifted:
        S = np.maximum(P[None, :, :], P[:, None, :])
        Dm = np.sqrt(((P[:, None, :] - S) ** 2).sum(-1))
    else:
        Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(Dm, np.inf)
    return Dm


def _feasible_prob(C, CM):
    lfp = np.ones(len(C))
    with np.errstate(all="ignore"):
        for j in range(C.shape[1]):
            lfp = np.fmin(lfp, norm_cdf(-C[:, j] / np.sqrt(CM[:, j])))
    return lfp


def nd_sort_cppd(F, FM, C, CM, n_sort):
    """Non-dominated sorting under probabilistic dominance weighted by the probability of feasibility; every front takes the
    solutions dominated by the fewest remaining ones."""
    N = len(F)
    lfp = _feasible_prob(C, CM)
    with np.errstate(all="ignore"):
        mean = F[:, None, :] - F[None, :, :]
        sig = np.sqrt(FM[:, None, :] + FM[None, :, :])
        x = norm_cdf(-mean / sig)
    y = 1 - x
    x = -x * lfp[:, None, None]
    y = -y * lfp[None, :, None]
    le, ge, eq = np.all(x <= y, -1), np.all(x >= y, -1), np.all(x == y, -1)
    iu = np.triu(np.ones((N, N), bool), 1)
    dom = np.zeros((N, N), bool)
    a = iu & le & ~eq
    b = iu & ~a & ge & ~eq
    dom[a] = True
    dom[b.T] = True
    front = np.full(N, np.inf)
    maxf = 0
    while np.sum(front != np.inf) < min(n_sort, N):
        maxf += 1
        cur = np.where(front == np.inf)[0]
        cnt = dom[np.ix_(cur, cur)].sum(0)
        idx = cur[cnt == cnt.min()]
        front[idx] = maxf
        dom[idx, :] = False
    return front, maxf


def _norm_cv(F, C):
    with np.errstate(all="ignore"):
        F = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 10e-10)
        C = np.maximum(C, 0)
        C = C / C.max(0) if C.shape[1] else C
        cv = C.sum(1)
        cv = cv / max(np.nanmax(cv) if cv.size and not np.all(np.isnan(cv)) else 0.0, 10e-10)
    return F, cv


def _selection(F, C, front, maxf, N, plus):
    nxt = front < maxf
    last = np.where(front == maxf)[0]
    if plus:
        F, cv = _norm_cv(F, C)
        P = np.column_stack([F, cv])
    else:
        P = F
    if maxf == 1:
        keep = ~truncate_lexi(_dist(P[last], plus), len(last) - N)
        nxt[last[keep]] = True
    else:
        n1 = np.where(nxt)[0]
        allp = np.concatenate([n1, last])
        Dm = _dist(P[allp], plus) if plus else _dist(P[allp], False)
        if not plus:
            np.fill_diagonal(Dm, 0.0)
        N1 = len(n1)
        sel1, sel2 = list(range(N1)), list(range(N1, len(allp)))
        for _ in range(N - N1):
            d = np.sort(Dm[np.ix_(sel2, sel1)], axis=1)[:, 0]
            k = _nanargmax(d)
            sel1.append(sel2.pop(k))
        nxt[allp[np.array(sel1[N1:], dtype=int)]] = True
    return nxt


class PEABase(LoopAlgorithm):
    PLUS = False

    def __init__(self, pop_size: int = 100, wmax: int = 20, mu: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax, self.mu = int(wmax), int(mu)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.NI = self.N
        P, _ = UniformPoint(self.NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills            # database
        self.P = infills
        nc = cons(infills).shape[1]
        self.th_obj, self.th_con = 5.0 * np.ones((self.M, self.D)), 5.0 * np.ones((nc, self.D))
        self.success, self.mo, self.mc = True, None, None
        self._set_optimum()

    def _train(self):
        X, F, C = decs(self.pop), objs(self.pop), cons(self.pop)
        D = self.D
        d1 = np.unique(np.round(X * 1e10) / 1e10, axis=0, return_index=True)[1]

        def fit(y, th):
            d2 = np.unique(np.round(y * 1e10) / 1e10, return_index=True)[1]
            dist = np.intersect1d(d1, d2)
            if len(dist) <= D + 1:
                # deviation: the reference stops ("least squares problem is underdetermined") when a response has too few
                # distinct values for the linear trend (e.g. a degenerate objective); keep every distinct design site instead
                dist = d1
            m = DaceModel(X[dist], y[dist], "regpoly1", th, 1e-5 * np.ones(D), 100 * np.ones(D))
            return m, m.theta

        self.mo, self.mc = [], []
        for i in range(F.shape[1]):
            m, self.th_obj[i] = fit(F[:, i], self.th_obj[i])
            self.mo.append(m)
        for i in range(C.shape[1]):
            m, self.th_con[i] = fit(C[:, i], self.th_con[i])
            self.mc.append(m)

    def _predict(self, X):
        po = [m.predict(X, mse=True) for m in self.mo]
        pc = [m.predict(X, mse=True) for m in self.mc]
        F = np.column_stack([p[0] for p in po])
        FM = np.column_stack([p[1] for p in po])
        C = np.column_stack([p[0] for p in pc]) if pc else np.zeros((len(X), 0))
        CM = np.column_stack([p[1] for p in pc]) if pc else np.zeros((len(X), 0))
        return F, FM, C, CM

    def _evo_search(self):
        X, F, C = decs(self.P), objs(self.P), cons(self.P)
        FM, CM = np.zeros_like(F), np.zeros_like(C)
        n = len(self.P)
        for _ in range(self.wmax):
            od = ga(self.problem, X, rng=self.rng)
            of, ofm, oc, ocm = self._predict(od)
            X, F, C = np.vstack([X, od]), np.vstack([F, of]), np.vstack([C, oc])
            FM, CM = np.vstack([FM, ofm]), np.vstack([CM, ocm])
            front, maxf = nd_sort_cppd(F, FM, C, CM, n)
            idx = np.where(_selection(F, C, front, maxf, n, self.PLUS))[0]
            X, F, C, FM, CM = X[idx], F[idx], C[idx], FM[idx], CM[idx]
        return X, F, C, FM, CM

    def _far_from_db(self, cand):
        db = decs(self.pop)
        keep = [c for c in cand if np.sqrt(((db - c) ** 2).sum(1)).min() > 1e-5]
        return np.array(keep).reshape(-1, self.D)

    def _candidates(self, X, F, C, FM, CM):
        mu, N = self.mu, self.N
        db = decs(self.pop)
        inside = np.array([np.any(np.all(db == x, axis=1)) for x in X])
        if inside.all():
            return np.zeros((0, self.D))
        if (~inside).sum() <= mu:
            return self._far_from_db(X[~inside])
        X, F, C, FM, CM = X[~inside], F[~inside], C[~inside], FM[~inside], CM[~inside]
        AF, AC = objs(self.pop), cons(self.pop)
        allf = np.vstack([AF, F])
        zmin, zmax = allf.min(0), allf.max(0)
        rng_ = np.maximum(zmax - zmin, 10e-10)
        AF, F, FM = (AF - zmin) / rng_, (F - zmin) / rng_, FM / rng_ ** 2
        if self.PLUS:
            with np.errstate(all="ignore"):
                C, AC = np.maximum(C, 0), np.maximum(AC, 0)
                if C.shape[1]:
                    C = C / np.vstack([C, AC]).max(0)
                    AC = AC / np.vstack([C, AC]).max(0)
                pcv, acv = C.sum(1), AC.sum(1)
                den = max(np.nanmax(np.concatenate([pcv, acv, [10e-10]])), 10e-10)
                pcv, acv = pcv / den, acv / den
        num = int(np.all(AC <= 0, axis=1).sum())
        fr, _ = nd_sort(AF, AC if AC.shape[1] else None, np.inf)
        if num >= N:
            sel = fr == 1
        else:
            sel, i = fr == 1, 1
            while sel.sum() < N:
                sel |= fr == i
                i += 1
        A2 = AF[sel]
        if self.PLUS:
            A2 = np.column_stack([A2, acv[sel]])
            PP = np.column_stack([F, pcv])
        else:
            PP = F
        front, maxf = nd_sort_cppd(F, FM, C, CM, mu)
        nxt = front < maxf
        last = list(np.where(front == maxf)[0])
        if len(last) == mu - nxt.sum():
            nxt[last] = True
        elif len(last) > mu - nxt.sum():
            A2 = np.vstack([A2, PP[nxt]])
            for _ in range(mu - int(nxt.sum())):
                L = PP[last]
                if self.PLUS:
                    S = np.maximum(A2[None, :, :], L[:, None, :])
                    d = np.sqrt(((L[:, None, :] - S) ** 2).sum(-1)).min(1)
                else:
                    d = np.sqrt(((L[:, None, :] - A2[None, :, :]) ** 2).sum(-1)).min(1)
                k = _nanargmax(d)
                nxt[last[k]] = True
                A2 = np.vstack([A2, PP[last[k]]])
                del last[k]
        return self._far_from_db(X[nxt])

    def step(self):
        if self.success:
            self._train()
        cand = self._candidates(*self._evo_search())
        self.success = False
        if len(cand):
            self.pop = Population.merge(self.pop, self.evaluate(cand))
            F, C = objs(self.pop), cons(self.pop)
            front, maxf = nd_sort(F, C if C.shape[1] else None, self.NI)
            self.P = self.pop[_selection(F, C, front, maxf, self.NI, self.PLUS)]
            self.success = True
