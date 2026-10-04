"""Kriging-assisted EA with decision-space density weighted probabilistic dominance and angle-based infill, plus a
reference-vector local search when an infill round brings no improvement (DISK / DISK+ for constrained problems)."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, truncate_lexi, uniform_point
from algorithms.community_utils.dace import DaceModel, norm_cdf
from algorithms.community_utils.pea import _nanargmax
from core.population import Population

__all__ = ["DISKBase"]


def _angles(A, B):
    with np.errstate(all="ignore"):
        c = (A @ B.T) / (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None])
    return np.arccos(np.clip(c, -1, 1))


def _norm(F):
    lo = F.min(0)
    return (F - lo) / np.maximum(F.max(0) - lo, 10e-10)


def _density(X, mu, K):
    """Multivariate normal density of the decision vectors (NaN when the covariance is singular)."""
    D = X.shape[1]
    try:
        det = np.linalg.det(K)
        Ki = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        return np.full(len(X), np.nan)
    d = X - mu
    with np.errstate(all="ignore"):
        return 1.0 / (np.sqrt(det) * (2 * np.pi) ** (D / 2)) * np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, Ki, d))


def nd_sort_dipd(X, F, FM, n_sort, mu, K, C=None):
    N = len(F)
    pro = _density(X, mu, K)
    i_idx, j_idx = np.triu_indices(N, 1)
    with np.errstate(all="ignore"):
        x = norm_cdf(-(F[i_idx] - F[j_idx]) / np.sqrt(FM[i_idx] + FM[j_idx]))
    y = 1 - x
    x, y = -x * pro[i_idx, None], -y * pro[j_idx, None]
    eq = np.all(x == y, 1)
    a = np.all(x <= y, 1) & ~eq
    b = ~a & np.all(x >= y, 1) & ~eq
    if C is not None:
        cv = np.maximum(0, C).sum(1)
        ci, cj = cv[i_idx], cv[j_idx]
        a = np.where(ci < cj, True, np.where(ci > cj, False, a & (ci == cj)))
        b = np.where(ci > cj, True, np.where(ci < cj, False, b & (ci == cj)))
    dom = np.zeros((N, N), bool)
    dom[i_idx[a], j_idx[a]] = True
    dom[j_idx[b], i_idx[b]] = True
    front, maxf = np.full(N, np.inf), 0
    while np.sum(front != np.inf) < min(n_sort, N):
        maxf += 1
        cur = np.where(front == np.inf)[0]
        cnt = dom[np.ix_(cur, cur)].sum(0)
        idx = cur[cnt == cnt.min()]
        front[idx] = maxf
        dom[idx, :] = False
    return front, maxf


def _survive(Fn, front, maxf, N):
    nxt = front < maxf
    last = np.where(front == maxf)[0]
    if maxf == 1:
        Dm = _angles(Fn[last], Fn[last])
        np.fill_diagonal(Dm, np.inf)
        nxt[last[~truncate_lexi(Dm, len(last) - N)]] = True
    else:
        n1 = np.where(nxt)[0]
        allp = np.concatenate([n1, last])
        Dm = _angles(Fn[allp], Fn[allp])
        np.fill_diagonal(Dm, np.inf)
        N1 = len(n1)
        s1, s2 = list(range(N1)), list(range(N1, len(allp)))
        for _ in range(N - N1):
            k = _nanargmax(np.sort(Dm[np.ix_(s2, s1)], 1)[:, 0])
            s1.append(s2.pop(k))
        nxt[allp[np.array(s1[N1:], dtype=int)]] = True
    return nxt


class DISKBase(LoopAlgorithm):
    PLUS = False

    def __init__(self, pop_size: int = 100, wmax: int = 60, alpha: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax, self.alpha = int(wmax), int(alpha)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.NI = self.N
        P, _ = UniformPoint(self.NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop, self.A1 = infills, infills
        nc = cons(infills).shape[1] if self.PLUS else 0
        self.th_obj, self.th_con = 5.0 * np.ones((self.M, self.D)), 5.0 * np.ones((nc, self.D))
        self._set_optimum()

    # -- models ------------------------------------------------------------
    def _fit(self, X, y, th, regr):
        D = self.D
        d1 = np.unique(X, axis=0, return_index=True)[1]
        d2 = np.unique(y, return_index=True)[1]
        dist = np.intersect1d(d1, d2)
        if regr == "regpoly1" and len(dist) <= D + 1:
            dist = d1      # deviation (see community_utils/pea.py)
        m = DaceModel(X[dist], y[dist], regr, th, 1e-5 * np.ones(D), 100 * np.ones(D))
        return m, m.theta

    def _train(self):
        X, F = decs(self.pop), objs(self.pop)
        self.mo, self.mc = [], []
        for i in range(self.M):
            m, self.th_obj[i] = self._fit(X, F[:, i], self.th_obj[i], "regpoly1")
            self.mo.append(m)
        if self.PLUS:
            C = cons(self.pop)
            for i in range(C.shape[1]):
                m, self.th_con[i] = self._fit(X, C[:, i], self.th_con[i], "regpoly0")
                self.mc.append(m)

    def _predict(self, X):
        po = [m.predict(X, mse=True) for m in self.mo]
        F, FM = np.column_stack([p[0] for p in po]), np.abs(np.column_stack([p[1] for p in po]))
        if self.mc:
            pc = [m.predict(X, mse=True) for m in self.mc]
            C, CM = np.column_stack([p[0] for p in pc]), np.abs(np.column_stack([p[1] for p in pc]))
        else:
            C, CM = np.zeros((len(X), 0)), np.zeros((len(X), 0))
        return F, FM, C, CM

    def _sort(self, X, F, FM, C, n):
        return nd_sort_dipd(X, F, FM, n, self.mu_, self.K_, C if self.PLUS else None)

    # -- phases ------------------------------------------------------------
    def _optimise(self):
        X = decs(self.A1)
        N = len(X)
        for _ in range(self.wmax):
            X = np.vstack([X, ga(self.problem, X, rng=self.rng)])
            F, FM, C, CM = self._predict(X)
            front, maxf = self._sort(X, F, FM, C, N)
            idx = np.where(_survive(_norm(F), front, maxf, N))[0]
            X, F, FM, C, CM = X[idx], F[idx], FM[idx], C[idx], CM[idx]
        return X, F, FM, C, CM

    def _new_select(self, X, F, FM, C, CM):
        db = decs(self.pop)
        idx = [i for i in range(len(X)) if np.sqrt(((db - X[i]) ** 2).sum(1)).min() > 1e-50]
        if len(idx) <= self.alpha:
            return self.evaluate(X[idx]) if idx else None
        X, F, FM, C, CM = X[idx], F[idx], FM[idx], C[idx], CM[idx]
        AF, AC = objs(self.pop), cons(self.pop)
        if self.PLUS:
            A2 = AF
        else:
            f, _ = nd_sort(AF, None, 1)
            A2 = np.unique(AF[f == 1], axis=0)
        allf = np.vstack([A2, F])
        zmin, r = allf.min(0), np.maximum(allf.max(0) - allf.min(0), 10e-10)
        A2, F, FM = (A2 - zmin) / r, (F - zmin) / r, FM / r ** 2
        if self.PLUS:
            num = int(np.all(AC <= 0, 1).sum())
            fr, _ = nd_sort(A2, AC if AC.shape[1] else None, np.inf)
            if num >= self.N:
                A2 = A2[fr == 1]
            else:
                sel, i = fr == 1, 1
                while sel.sum() < self.N:
                    sel |= fr == i
                    i += 1
                A2 = A2[sel]
        front, _ = self._sort(X, F, FM, C, 1)
        X, F = X[front == 1], F[front == 1]
        if len(X) <= self.alpha:
            return self.evaluate(X)
        alive = np.ones(len(X), bool)
        out = None
        while (~alive).sum() < self.alpha:
            last = np.where(alive)[0]
            dis = np.sort(_angles(F[last], A2), 1)[:, 0]
            k = last[_nanargmax(dis)]
            new = self.evaluate(X[[k]])
            out = new if out is None else Population.merge(out, new)
            if self.PLUS:
                A2 = np.vstack([A2, (objs(out) - zmin) / r])
            else:
                allo = np.vstack([objs(self.pop), objs(out)])
                f, _ = nd_sort(allo, None, 1)
                A2 = (np.unique(allo[f == 1], axis=0) - zmin) / r
            alive[k] = False
        return out

    def _judge_ls(self, Cp):
        FC, FA = objs(Cp), objs(self.pop)
        if self.PLUS:
            CC, CA = cons(Cp), cons(self.pop)
            f1, _ = nd_sort(FC, CC if CC.shape[1] else None, 1)
            f2, _ = nd_sort(FA, CA if CA.shape[1] else None, 1)
            a, b = FC[f1 == 1], FA[f2 == 1]
            ca = np.maximum(0, CC[f1 == 1]).sum(1)
            cb = np.maximum(0, CA[f2 == 1]).sum(1)
            dom = np.all(a[:, None] <= b[None], -1) & ~np.all(a[:, None] == b[None], -1)
            dom = np.where(ca[:, None] == cb[None], dom, ca[:, None] < cb[None])
        else:
            f1, _ = nd_sort(FC, None, 1)
            f2, _ = nd_sort(FA, None, 1)
            a, b = FC[f1 == 1], FA[f2 == 1]
            dom = np.all(a[:, None] <= b[None], -1) & ~np.all(a[:, None] == b[None], -1)
        return 0 if dom.any() else 1

    def _identify_w(self):
        V, _ = uniform_point(10 * self.N, self.M)
        F = objs(self.pop)
        f, _ = nd_sort(F, None, 1)
        A = F[f == 1]
        nadir, ideal = A.max(0), A.min(0)
        ideal = ideal - (nadir - ideal) / 10 - 0.1
        mins = np.sort(_angles(V, A - ideal), 1)[:, 0]
        idx = np.where(mins == np.nanmax(mins))[0]
        return V[idx[int(self.rng.integers(0, len(idx)))] if len(idx) > 1 else idx[0]], ideal

    def _de(self, P, current):
        rng, (N, D) = self.rng, P.shape
        Fv = np.array([0.6, 0.8, 1.0])[rng.integers(0, 3, N)][:, None] * np.ones((1, D))
        CR = np.array([0.1, 0.2, 1.0])[rng.integers(0, 3, N)][:, None]
        site = rng.random((N, D)) < CR
        P1, P2, P3 = P[rng.permutation(N)], P[rng.permutation(N)], P[rng.permutation(N)]
        off = P.copy()
        if current:
            off[site] = P[site] + Fv[site] * (P1[site] - P[site]) + Fv[site] * (P2[site] - P3[site])
        else:
            off[site] = P1[site] + Fv[site] * (P2[site] - P3[site])
        lo, up = np.broadcast_to(self.lower, off.shape), np.broadcast_to(self.upper, off.shape)
        s = rng.random((N, D)) < 1.0 / D
        m = rng.random((N, D))
        off = np.minimum(np.maximum(off, lo), up)
        with np.errstate(all="ignore"):
            t = s & (m <= 0.5)
            off[t] = off[t] + (up[t] - lo[t]) * ((2 * m[t] + (1 - 2 * m[t]) * (1 - (off[t] - lo[t]) / (up[t] - lo[t])) ** 21) ** (1 / 21) - 1)
            t = s & (m > 0.5)
            off[t] = off[t] + (up[t] - lo[t]) * (1 - (2 * (1 - m[t]) + 2 * (m[t] - 0.5) * (1 - (up[t] - off[t]) / (up[t] - lo[t])) ** 21) ** (1 / 21))
        return off

    def _ls_fitness(self, F, FM, C, CM, W, ideal):
        fit = np.max(np.abs(F - ideal) * W, 1)
        if self.PLUS:
            fit = fit - 2 * (np.sqrt(FM).mean(1) + (np.sqrt(CM).mean(1) if CM.shape[1] else 0.0)) / 2
            cv = np.maximum(0, C).sum(1)
            inf = cv > 0
            fit[inf] = fit.max() + cv[inf]
        else:
            fit = fit - 2 * np.sqrt(FM).mean(1)
        return fit

    def _local_search(self, X0, W, ideal):
        X = X0
        N = len(X)
        for _ in range(self.wmax):
            X = np.unique(np.vstack([X, ga(self.problem, X, rng=self.rng), self._de(X, True), self._de(X, False), self._de(X, True)]), axis=0)
            F, FM, C, CM = self._predict(X)
            r = np.argsort(self._ls_fitness(F, FM, C, CM, W, ideal), kind="stable")[:N]
            X = X[r]
        F, FM, C, CM = self._predict(X)
        best = X[int(np.argsort(self._ls_fitness(F, FM, C, CM, W, ideal), kind="stable")[0])]
        if np.sqrt(((decs(self.pop) - best) ** 2).sum(1)).min() > 1e-50:
            self.pop = Population.merge(self.pop, self.evaluate(best[None]))

    def step(self):
        self._train()
        F, X = objs(self.pop), decs(self.pop)
        C = cons(self.pop)
        use_c = self.PLUS and np.any(np.maximum(0, C).sum(1) != 0)
        fr, _ = nd_sort(F, C if use_c else None, np.inf)
        P = X[fr == 1]
        if len(P) <= 1:
            P = np.vstack([P, X[fr == 2]])
        self.mu_ = P.mean(0)
        d = P - self.mu_
        self.K_ = d.T @ d / (len(P) - 1)
        OP = self._optimise()
        Cp = self._new_select(*OP)
        flag = 0
        if Cp is not None and len(Cp):
            flag = self._judge_ls(Cp)
            self.pop = Population.merge(self.pop, Cp)
        if flag == 1:
            self._train()
            W, ideal = self._identify_w()
            self._local_search(OP[0], W, ideal)
        F, C = objs(self.pop), cons(self.pop)
        Fn = _norm(F)
        front, maxf = nd_sort(Fn, C if (self.PLUS and C.shape[1]) else None, self.NI)
        self.A1 = self.pop[_survive(Fn, front, maxf, self.NI)]
