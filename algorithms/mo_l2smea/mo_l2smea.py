# emopylab 2026
"""MO-L2SMEA (multi-objective linear subspace surrogate modeling assisted evolutionary algorithm).

Reference:
L. Si, X. Zhang, Y. Tian, S. Yang, L. Zhang, and Y. Jin. Linear subspace surrogate modeling for
large-scale expensive single/multi-objective optimization. IEEE Transactions on Evolutionary
Computation, 2025, 29(3): 697-710.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, objs, tournament, uniform_point
from algorithms.community_utils.dace import DaceModel
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population

ALGORITHM_FLAGS = {'MOL2SMEA': {'expensive', 'integer', 'multi', 'real'}}


def _cos_dist(A, B):
    with np.errstate(all="ignore"):
        c = (A @ B.T) / (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None])
    return 1 - c


def _boundary(start, direct):
    D = len(start)
    lmax = np.sqrt(D) + 1e-8

    def search(le):
        ls = 0.0
        while abs(le - ls) >= 1e-12:
            mid = (ls + le) / 2
            x = start + mid * direct
            if np.any((x > 1) | (x < 0)):
                le = mid
            else:
                ls = mid
        return ls

    return search(lmax), search(-lmax)


class _Line:
    """1-D Kriging subproblem along a line of the normalised decision space (fixed theta = n^-1, constant trend)."""

    def __init__(self, start, end, arc, D, fmin, fmax, w):
        self.start = start
        d = end - start
        self.direct = d / np.sqrt((d ** 2).sum())
        self.ub, self.lb = _boundary(self.start, self.direct)
        X, F = arc[:, :D], arc[:, D:]
        with np.errstate(all="ignore"):
            fit = np.max(np.abs(F - fmin) / (fmax - fmin) * w, 1)
        dist = np.sqrt(((X - start) ** 2).sum(1))
        v2 = X - start
        mvl = (v2 ** 2).sum(1)
        mvl[mvl == 0] = np.inf
        cosv = (v2 @ self.direct) / np.sqrt(mvl)
        sinv = np.sqrt(np.maximum(1 - cosv ** 2, 0))
        perp, newx = sinv * dist, cosv * dist
        alpha = np.sort(perp)[min(10, len(perp)) - 1]
        sel = perp <= alpha
        tx, ty = newx[sel], fit[sel]
        # deviation: near-identical positions (repeated archive points, differences ~1e-17) would make the fit fail
        u = np.unique(np.round(tx, 12), return_index=True)[1]
        tx, ty = tx[u], ty[u]
        self.train_dec = tx
        self.model = DaceModel(tx[:, None], ty, "regpoly0", np.array([len(tx) ** -1.0]))

    def predict(self, t):
        y, v = self.model.predict(np.asarray(t, float)[:, None], mse=True)
        return y, v


def _last_selection(F1, F2, K, Z, zmin, rng):
    F = np.vstack([F1, F2]) - zmin
    N, M = F.shape
    N1 = len(F1)
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    ext = [int(np.argmin(np.max(F / w[i], 1))) for i in range(M)]
    with np.errstate(all="ignore"):
        F = F / F[ext].max(0)
    cos = 1 - _cos_dist(F, Z)
    dist = np.linalg.norm(F, axis=1)[:, None] * np.sqrt(np.maximum(1 - cos ** 2, 0))
    dist = np.where(np.isnan(dist), np.inf, dist)
    pi = np.argmin(dist, 1)
    d = dist[np.arange(N), pi]
    rho = np.bincount(pi[:N1], minlength=len(Z)).astype(float)
    choose = np.zeros(len(F2), bool)
    zc = np.ones(len(Z), bool)
    while choose.sum() < K:
        tmp = np.where(zc)[0]
        jm = np.where(rho[tmp] == rho[tmp].min())[0]
        j = tmp[jm[rng.integers(0, len(jm))]]
        I = np.where(~choose & (pi[N1:] == j))[0]
        if len(I):
            s = int(np.argmin(d[N1 + I])) if rho[j] == 0 else int(rng.integers(0, len(I)))
            choose[I[s]] = True
            rho[j] += 1
        else:
            zc[j] = False
    return choose


class MOL2SMEA(LoopAlgorithm):
    """For each of five objective-space clusters a CMA-ES distribution samples ``2*NLinear`` points that define ``NLinear``
    lines in decision space; along each line a 1-D Kriging model of the cluster's Tchebycheff value is built from the
    nearest archive points, NSGA-III searches the joint line positions, and the promising positions far from the training
    positions are evaluated; the CMA-ES mean and covariance then move towards the cluster's non-dominated solutions."""

    def __init__(self, pop_size: int = 100, NLinear: int = 8, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.NL = int(NLinear)

    def _initialize_infill(self):
        self.NT = 2 * self.D
        return self.evaluate(self.lower + lhs_design(self.rng, self.NT, self.D) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        D, M, NL = self.D, self.M, self.NL
        self.pop = infills
        F = objs(infills)
        self.fmax, self.fmin = F.max(0), F.min(0)
        Xn = (decs(infills) - self.lower) / (self.upper - self.lower)
        self.arc = np.hstack([Xn, F])
        self.W, self.C = uniform_point(5, M)
        grp = self._groups()
        self.xm, self.sig, self.pc, self.ps, self.B, self.Dg, self.Cv, self.isc = [], [], [], [], [], [], [], []
        for k in range(self.C):
            self.xm.append(Xn[grp == k].mean(0) if (grp == k).any() else Xn.mean(0))
            self.sig.append(0.5); self.pc.append(np.zeros(D)); self.ps.append(np.zeros(D))
            self.B.append(np.eye(D)); self.Dg.append(np.ones(D)); self.Cv.append(np.eye(D)); self.isc.append(np.eye(D))
        self.eig_eval = 0
        self.chiD = D ** 0.5 * (1 - 1 / (4 * D) + 1 / 21 * D ** 2)      # literal: D^2 (not 1/(21 D^2))
        self.lam = 4 * NL
        mu = self.lam / 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, int(np.floor(mu)) + 1))
        self.mu = int(np.floor(mu))
        self.w = w / w.sum()
        self.mueff = self.w.sum() ** 2 / (self.w ** 2).sum()
        me = self.mueff
        self.cc = (4 + me / D) / (D + 4 + 2 * me / D)
        self.cs = (me + 2) / (D + me + 5)
        self.c1 = 2 / ((D + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((D + 2) ** 2 + me))
        self.damps = 1 + 2 * max(0, np.sqrt((me - 1) / (D + 1)) - 1)
        self.cur = 0
        self.maxFE_l = self.max_FE - self.NT
        self._set_optimum()

    def _nobj(self):
        with np.errstate(all="ignore"):
            return (self.arc[:, self.D:] - self.fmin) / (self.fmax - self.fmin)

    def _groups(self):
        cd = _cos_dist(self.W, self._nobj())
        return np.argmin(np.where(np.isnan(cd), np.inf, cd), 0)

    def _nsga3(self, lines, pop_size=40, iters=30):
        rng = self.rng
        n = len(lines)
        BU = np.array([l.ub for l in lines])
        BD = np.array([l.lb for l in lines])
        Z, N = uniform_point(pop_size, n)
        ev = lambda X: np.column_stack([lines[k].predict(X[:, k])[0] for k in range(n)])
        X = rng.random((pop_size, n)) * (BU - BD) + BD
        F = ev(X)
        zmin = F.min(0)
        for _ in range(iters):
            mate = tournament(2, N, F.sum(1), rng=rng)
            P = X[mate]
            h = len(P) // 2
            A, Bp = P[:h], P[h:2 * h]
            mu_ = rng.random(A.shape)
            beta = np.where(mu_ <= 0.5, (2 * mu_) ** (1 / 21), (2 - 2 * mu_) ** (-1 / 21))
            beta = beta * (-1.0) ** rng.integers(0, 2, A.shape)
            beta[rng.random(A.shape) < 0.5] = 1
            O = np.vstack([(A + Bp) / 2 + beta * (A - Bp) / 2, (A + Bp) / 2 - beta * (A - Bp) / 2])
            lo, up = np.broadcast_to(BD, O.shape), np.broadcast_to(BU, O.shape)
            s, m = rng.random(O.shape) < 1 / n, rng.random(O.shape)
            O = np.minimum(np.maximum(O, lo), up)
            with np.errstate(all="ignore"):
                t = s & (m <= 0.5)
                O[t] = O[t] + (up[t] - lo[t]) * ((2 * m[t] + (1 - 2 * m[t]) * (1 - (O[t] - lo[t]) / (up[t] - lo[t])) ** 21) ** (1 / 21) - 1)
                t = s & (m > 0.5)
                O[t] = O[t] + (up[t] - lo[t]) * (1 - (2 * (1 - m[t]) + 2 * (m[t] - 0.5) * (1 - (up[t] - O[t]) / (up[t] - lo[t])) ** 21) ** (1 / 21))
            OF = ev(O)
            zmin = np.minimum(zmin, OF.min(0))
            AX, AF = np.vstack([X, O]), np.vstack([F, OF])
            front, maxf = nd_sort(AF, None, N)
            nxt = front < maxf
            last = np.where(front == maxf)[0]
            nxt[last[_last_selection(AF[nxt], AF[last], N - int(nxt.sum()), Z, zmin, rng)]] = True
            X, F = AX[nxt], AF[nxt]
        V = np.column_stack([lines[k].predict(X[:, k])[1] for k in range(n)])
        return X, F, V

    def _select(self, lines, X, F, V):
        D, n = self.D, len(lines)
        # literal: the reference sorts an empty column range, so every candidate is in the first front
        new = []
        for k, l in enumerate(lines):
            th = 0.07 * (l.ub - l.lb) * n
            md = np.abs(X[:, [k]] - l.train_dec[None]).min(1)
            rem = md >= th
            rd, rf, rs = X[rem, k], F[rem, k], -V[rem, k]
            if rem.sum() > 2:
                sel = np.arange(int(rem.sum()))
                fr, _ = nd_sort(np.column_stack([rs, rf]), None, 1)
                while np.any(fr > 1) and len(sel) > 2:
                    sel = sel[fr == 1]
                    fr, _ = nd_sort(np.column_stack([rs[sel], rf[sel]]), None, 1)
                pd, pf = rd[sel], rf[sel]
            elif rem.sum() >= 1:
                pd, pf = rd, rf
            else:
                continue
            new.append(np.column_stack([l.start + pd[:, None] * l.direct, pf]))
        if new:
            return np.vstack(new)
        MI = np.argmin(F, 0)
        LI = int(np.argmin(F[MI, np.arange(n)]))
        if len(X) == 1:
            LI = int(MI[0]) if np.ndim(MI) else 0
            pos, f = X[0, LI], F[0, LI]
        else:
            pos, f = X[MI[LI], LI], F[MI[LI], LI]
        return np.concatenate([lines[LI].start + pos * lines[LI].direct, [f]])[None]

    def _update(self, off, tarc):
        D = self.D
        remain = self.maxFE_l - self.cur
        if len(off) == 0 or remain <= 0:
            return tarc
        if len(off) > remain:
            off = off[np.argsort(off[:, D], kind="stable")[:remain]]
        new = self.evaluate(self.lower + off[:, :D] * (self.upper - self.lower))
        self.pop = Population.merge(self.pop, new)
        Xn = (decs(new) - self.lower) / (self.upper - self.lower)
        F = objs(new)
        self.fmin, self.fmax = np.minimum(self.fmin, F.min(0)), np.maximum(self.fmax, F.max(0))
        rows = np.hstack([Xn, F])
        self.arc = np.vstack([self.arc, rows])
        self.cur += len(off)
        return np.vstack([tarc, rows])

    def step(self):
        rng, D, NL = self.rng, self.D, self.NL
        grp = self._groups()
        nobj0 = self._nobj()                 # NObjs / grp are computed once per generation in the reference
        self._nobj0 = nobj0
        for k in range(self.C):
            refs = np.array([self.xm[k] + self.sig[k] * self.B[k] @ (self.Dg[k] * rng.standard_normal(D)) for _ in range(2 * NL)])
            refs[refs < 0], refs[refs > 1] = 1e-6, 1 - 1e-6
            mask = np.zeros(len(self.arc), bool)
            mask[: len(grp)] = grp == k             # a shorter logical mask selects among the first rows
            tarc = self.arc[mask]
            if (grp == k).sum() < 30:
                cd = _cos_dist(self.W[[k]], nobj0)[0]
                tarc = self.arc[np.argsort(np.where(np.isnan(cd), np.inf, cd), kind="stable")[: 2 * D]]
            lines = [_Line(refs[2 * j], refs[2 * j + 1], tarc, D, self.fmin, self.fmax, self.W[k]) for j in range(NL)]
            X, F, V = self._nsga3(lines)
            off = self._select(lines, X, F, V)
            tarc = self._update(off, tarc)
            self._cma(k, tarc, grp)

    def _cma(self, k, tarc, grp):
        rng, D, mu = self.rng, self.D, self.mu
        fr, _ = nd_sort(tarc[:, D:], None, len(tarc))
        if len(tarc) >= mu:
            i_n, i_d = np.where(fr == 1)[0], np.where(fr != 1)[0]
            h = int(np.ceil(mu / 2))
            if len(i_n) >= h:
                if len(i_d) >= h:
                    md = np.vstack([tarc[i_n[rng.permutation(len(i_n))[:h]], :D], tarc[i_d[rng.permutation(len(i_d))[:h]], :D]])
                else:
                    md = np.vstack([tarc[i_d, :D], tarc[i_n[rng.permutation(len(i_n))[: mu - len(i_d)]], :D]])
            else:
                md = np.vstack([tarc[i_n, :D], tarc[i_d[rng.permutation(len(i_d))[: mu - len(i_n)]], :D]])
        else:
            cd = _cos_dist(self.W[[k]], self._nobj0)[0]
            md = self.arc[np.argsort(np.where(np.isnan(cd), np.inf, cd), kind="stable")[:mu], :D]
        md = md[:mu]
        w = self.w[: len(md)] / self.w[: len(md)].sum() if len(md) < mu else self.w
        old = self.xm[k]
        self.xm[k] = md.T @ w
        sg = self.sig[k]
        self.ps[k] = (1 - self.cs) * self.ps[k] + np.sqrt(self.cs * (2 - self.cs)) * self.mueff * self.isc[k] @ (self.xm[k] - old) / sg
        with np.errstate(all="ignore"):
            hsig = np.linalg.norm(self.ps[k]) / np.sqrt(1 - (1 - self.cs) ** (2 * self.cur / self.lam)) / self.chiD < 1.4 + 2 / (D + 1)
        self.pc[k] = (1 - self.cc) * self.pc[k] + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.xm[k] - old) / sg
        art = (md - old).T / sg
        self.Cv[k] = ((1 - self.c1 - self.cmu) * self.Cv[k] + self.c1 * (np.outer(self.pc[k], self.pc[k]) + (1 - hsig) * self.cc * (2 - self.cc) * self.Cv[k])
                      + self.cmu * art @ np.diag(w) @ art.T)
        self.sig[k] = sg * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps[k]) / self.chiD - 1))
        if self.cur - self.eig_eval > self.lam / (self.c1 + self.cmu) / D / 10:
            self.eig_eval = self.cur
            C = np.triu(self.Cv[k]) + np.triu(self.Cv[k], 1).T
            self.Cv[k] = C
            ev, B = np.linalg.eigh(C)
            dg = np.sqrt(np.maximum(ev, 0)).real
            self.B[k], self.Dg[k] = B, dg
            with np.errstate(divide="ignore"):
                self.isc[k] = B @ np.diag(1 / dg) @ B.T
