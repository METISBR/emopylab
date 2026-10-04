"""Kriging with the DACE recipe (maximum-likelihood correlation parameters found by the box-constrained pattern search of the
original toolbox), used by the expensive-optimisation ports.

``DaceModel(S, Y, regr, corr_theta0, lob, upb)`` fits a Gaussian-correlation Kriging model with a polynomial trend of order 0,
1 or 2; ``predict(X)`` returns the prediction and (optionally) the mean squared error."""

from __future__ import annotations

import numpy as np

try:  # scipy is a declared dependency; the numpy fallbacks keep the module self-contained
    from scipy.linalg import solve_triangular as _solve_tri
except Exception:  # pragma: no cover
    def _solve_tri(A, b, lower=True, trans=0):
        M = A.T if trans else A
        return np.linalg.solve(M, b)

__all__ = ["DaceModel", "norm_cdf", "norm_pdf"]

_EPS = np.finfo(float).eps


def norm_cdf(x):
    """Standard normal CDF (vectorised)."""
    try:
        from scipy.special import ndtr
        return ndtr(np.asarray(x, dtype=float))
    except Exception:  # pragma: no cover
        from math import erf
        x = np.asarray(x, dtype=float)
        return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0))) if x.ndim else 0.5 * (1.0 + erf(float(x) / np.sqrt(2.0)))


def norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def _regr(kind, S):
    S = np.atleast_2d(S)
    m, n = S.shape
    if kind == "regpoly0":
        return np.ones((m, 1))
    if kind == "regpoly1":
        return np.hstack([np.ones((m, 1)), S])
    cols = [np.ones(m)] + [S[:, i] for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            cols.append(S[:, i] * S[:, j])
    return np.column_stack(cols)


def _corrgauss(theta, d):
    return np.exp(np.sum(-(d ** 2) * np.reshape(theta, (1, -1)), axis=1))


class DaceModel:
    def __init__(self, S, Y, regr="regpoly1", theta0=None, lob=None, upb=None):
        S = np.atleast_2d(np.asarray(S, dtype=float))
        Y = np.asarray(Y, dtype=float).reshape(len(S), -1)
        m, n = S.shape
        self.regr_kind = regr
        mS, sS = S.mean(axis=0), S.std(axis=0, ddof=1) if m > 1 else np.zeros(n)
        mY, sY = Y.mean(axis=0), Y.std(axis=0, ddof=1) if m > 1 else np.zeros(Y.shape[1])
        sS = np.where(sS == 0, 1.0, sS)
        sY = np.where(sY == 0, 1.0, sY)
        Sn, Yn = (S - mS) / sS, (Y - mY) / sY
        ij = np.array([(k, l) for k in range(m - 1) for l in range(k + 1, m)], dtype=int).reshape(-1, 2)
        D = Sn[ij[:, 0]] - Sn[ij[:, 1]] if len(ij) else np.zeros((0, n))
        if len(D) and np.min(np.sum(np.abs(D), axis=1)) == 0:
            raise ValueError("Multiple design sites are not allowed")
        F = _regr(regr, Sn)
        if F.shape[1] > m:
            raise ValueError("least squares problem is underdetermined")
        self._par = dict(y=Yn, F=F, D=D, ij=ij, m=m)
        theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
        if lob is None:
            theta, f, fit = theta0, *self._objfunc(theta0)
        else:
            theta, f, fit = self._boxmin(theta0, np.asarray(lob, float), np.asarray(upb, float))
        if np.isinf(f):
            raise ValueError("Bad parameter region. Try increasing upb")
        self.theta, self.beta, self.gamma = np.asarray(theta, dtype=float), fit["beta"], fit["gamma"]
        self.sigma2 = sY ** 2 * fit["sigma2"]
        self.S, self.Ssc, self.Ysc = Sn, np.vstack([mS, sS]), np.vstack([mY, sY])
        self.C, self.Ft, self.G = fit["C"], fit["Ft"], fit["G"]

    # -- likelihood ----------------------------------------------------------------------------------------
    def _objfunc(self, theta):
        p = self._par
        m = p["m"]
        fit = None
        r = _corrgauss(theta, p["D"])
        R = np.eye(m) * (1.0 + (10 + m) * _EPS)
        idx = r > 0
        R[p["ij"][idx, 0], p["ij"][idx, 1]] = r[idx]
        R[p["ij"][idx, 1], p["ij"][idx, 0]] = r[idx]
        try:
            C = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            return np.inf, fit
        Ft = _solve_tri(C, p["F"], lower=True)
        Q, Gu = np.linalg.qr(Ft)
        if 1.0 / np.linalg.cond(Gu, 1) < 1e-10:
            if np.linalg.cond(p["F"]) > 1e15:
                raise ValueError("F is too ill conditioned")
            return np.inf, fit
        Yt = _solve_tri(C, p["y"], lower=True)
        beta = np.linalg.solve(Gu, Q.T @ Yt)
        rho = Yt - Ft @ beta
        sigma2 = np.sum(rho ** 2, axis=0) / m
        det_r = np.prod(np.diag(C) ** (2.0 / m))
        obj = float(np.sum(sigma2) * det_r)
        gamma = _solve_tri(C, rho, lower=True, trans=1).T          # rho' / C
        return obj, dict(sigma2=sigma2, beta=beta, gamma=gamma, C=C, Ft=Ft, G=Gu.T)

    # -- box-constrained pattern search of the original toolbox --------------------------------------------------
    def _boxmin(self, t0, lo, up):
        t, f, fit, it = self._start(t0, lo, up)
        if not np.isinf(f):
            p = len(t)
            kmax = 2 if p <= 2 else min(p, 4)
            for _ in range(kmax):
                th = t.copy()
                t, f, fit, it = self._explore(t, f, fit, it)
                t, f, fit, it = self._move(th, t, f, fit, it)
        return t, f, fit

    def _start(self, t0, lo, up):
        t = t0.astype(float).copy().reshape(-1)
        lo, up = lo.reshape(-1), up.reshape(-1)
        p = len(t)
        D = 2.0 ** (np.arange(1, p + 1) / (p + 2))
        ee = np.where(up == lo)[0]
        if len(ee):
            D[ee] = 1.0
            t[ee] = up[ee]
        ng = np.where((t < lo) | (up < t))[0]
        if len(ng):
            t[ng] = (lo[ng] * up[ng] ** 7) ** (1 / 8)
        ne = np.where(D != 1)[0]
        f, fit = self._objfunc(t)
        it = dict(D=D, ne=ne, lo=lo, up=up)
        if np.isinf(f):
            return t, f, fit, it
        if len(ng) > 1:
            d0, d1, q = 16.0, 2.0, len(ng)
            th, fh, jdom = t.copy(), f, ng[0]
            for k in range(q):
                j = ng[k]
                fk, tk = fh, th.copy()
                DD = np.ones(p)
                DD[ng] = 1.0 / d1
                DD[j] = 1.0 / d0
                alpha = np.min(np.log(lo[ng] / th[ng]) / np.log(DD[ng])) / 5
                v = DD ** alpha
                for _ in range(4):
                    tt = tk * v
                    ff, fitt = self._objfunc(tt)
                    if ff <= fk:
                        tk, fk = tt, ff
                        if ff <= f:
                            t, f, fit, jdom = tt, ff, fitt, j
                    else:
                        break
            if jdom > 0:
                D[[0, jdom]] = D[[jdom, 0]]
                it["D"] = D
        return t, f, fit, it

    def _explore(self, t, f, fit, it):
        for j in it["ne"]:
            tt = t.copy()
            DD = it["D"][j]
            if t[j] == it["up"][j]:
                atbd = True
                tt[j] = t[j] / np.sqrt(DD)
            elif t[j] == it["lo"][j]:
                atbd = True
                tt[j] = t[j] * np.sqrt(DD)
            else:
                atbd = False
                tt[j] = min(it["up"][j], t[j] * DD)
            ff, fitt = self._objfunc(tt)
            if ff < f:
                t, f, fit = tt, ff, fitt
            elif not atbd:
                tt[j] = max(it["lo"][j], t[j] / DD)
                ff, fitt = self._objfunc(tt)
                if ff < f:
                    t, f, fit = tt, ff, fitt
        return t, f, fit, it

    def _move(self, th, t, f, fit, it):
        p = len(t)
        v = t / th
        if np.all(v == 1):
            it["D"] = it["D"][np.r_[1:p, 0]] ** 0.2
            return t, f, fit, it
        rept = True
        while rept:
            tt = np.minimum(it["up"], np.maximum(it["lo"], t * v))
            ff, fitt = self._objfunc(tt)
            if ff < f:
                t, f, fit = tt, ff, fitt
                v = v ** 2
            else:
                rept = False
            if np.any((tt == it["lo"]) | (tt == it["up"])):
                rept = False
        it["D"] = it["D"][np.r_[1:p, 0]] ** 0.25
        return t, f, fit, it

    # -- prediction ----------------------------------------------------------------------------------------------
    def predict(self, X, mse: bool = False):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xn = (X - self.Ssc[0]) / self.Ssc[1]
        mx, m = len(Xn), len(self.S)
        dx = (Xn[:, None, :] - self.S[None, :, :]).reshape(mx * m, -1)
        f = _regr(self.regr_kind, Xn)
        r = _corrgauss(self.theta, dx).reshape(mx, m).T             # (m, mx)
        sy = f @ self.beta + (self.gamma @ r).T
        y = self.Ysc[0] + self.Ysc[1] * sy
        y = y[:, 0] if y.shape[1] == 1 else y
        if not mse:
            return y
        rt = _solve_tri(self.C, r, lower=True)
        u = np.linalg.solve(self.G, self.Ft.T @ rt - f.T)
        err = self.sigma2[0] * (1 + np.sum(u ** 2, axis=0) - np.sum(rt ** 2, axis=0))
        return y, err
