# emopylab 2026
"""OSP-NSDE (non-dominated sorting differential evolution with prediction in the objective space).

Reference:
E. Guerrero-Pena and A. F. R. Araujo. Multi-objective evolutionary algorithm with prediction in the
objective space. Information Sciences, 2019, 501: 293-316.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln

from algorithms.community_utils.base import LoopAlgorithm, cons, de, decs, nd_sort, objs, truncate_lexi
from core.population import Population
from util.hv import hypervolume

ALGORITHM_FLAGS = {'OSPNSDE': {'constrained', 'integer', 'multi', 'real'}}


def _hv(F, nadir):
    """Hypervolume of ``F`` after normalising by min(0, min F) and 1.1 x (nadir - that minimum), reference point 1."""
    if len(F) == 0:
        return 0.0
    fmin = np.minimum(F.min(0), 0)
    with np.errstate(all="ignore"):
        P = (F - fmin) / ((nadir - fmin) * 1.1)
    P = P[~np.any(P > 1, 1)]
    return float(hypervolume(P, np.ones(F.shape[1]))) if len(P) else 0.0


def ar1_forecast(series_fit, series_list, p):
    """Per-output AR(1) without constant (ARX with na = I, no inputs) fitted by least squares on ``series_fit`` (or on each
    series itself when it is None); returns the p-step-ahead forecast of the last value of every series."""
    def coef(y):
        y0, y1 = y[:-1], y[1:]
        den = (y0 ** 2).sum(0)
        with np.errstate(all="ignore"):
            a = np.where(den > 0, (y0 * y1).sum(0) / den, 1.0)
        return a

    out = []
    a_shared = coef(series_fit) if series_fit is not None else None
    for y in series_list:
        a = a_shared if a_shared is not None else coef(y)
        out.append(a ** p * y[-1])
    return np.array(out)


class VBGMM:
    """Variational Bayesian Gaussian mixture (Dirichlet / Normal-Wishart priors) with ``K`` components started from a random
    hard assignment; iterated until the relative change of the lower bound is below 1e-8 (at most 2000 iterations)."""

    def __init__(self, X, K, rng, tol=1e-8, max_iter=2000):
        X = np.asarray(X, float).T                           # d x n
        d, n = X.shape
        self.prior = dict(alpha=0.1, kappa=0.001, m=X.mean(1), v=float(d), M=1e-5 * np.eye(d))
        self.prior["logW"] = -2 * np.sum(np.log(np.diag(np.linalg.cholesky(self.prior["M"]).T)))
        lab = np.minimum((np.ceil(K * rng.random(n)) - 1).astype(int), K - 1)
        R = np.zeros((n, K))
        R[np.arange(n), lab] = 1
        self.R = R
        self._maximize(X)
        LB = -np.inf
        for _ in range(2, max_iter + 1):
            self._expect(X)
            self._maximize(X)
            lb = self._bound(X) / n
            if abs(lb - LB) < tol * abs(lb):
                break
            LB = lb
        for i in range(K):
            S = self.U[:, :, i]
            try:
                np.linalg.cholesky(S.T @ S)
            except np.linalg.LinAlgError:
                self.U[:, :, i] = (S + S.T) / 2

    def _maximize(self, X):
        pr, R = self.prior, self.R
        m = pr["kappa"] * pr["m"][:, None] + X @ R
        k = m.shape[1]
        d = X.shape[0]
        r = np.sqrt(R.T)
        U = np.zeros((d, d, k))
        logW = np.zeros(k)
        for i in range(k):
            Xm = (X - m[:, [i]]) * r[i]
            pm = pr["m"] - m[:, i]
            A = pr["M"] + Xm @ Xm.T + pr["kappa"] * np.outer(pm, pm)
            U[:, :, i] = np.linalg.cholesky((A + A.T) / 2).T     # upper triangular, A = U'U
            logW[i] = -2 * np.sum(np.log(np.diag(U[:, :, i])))
        self.alpha = pr["alpha"] + R.sum(0)
        self.kappa = pr["kappa"] + R.sum(0)
        self.mean = m / self.kappa
        self.v = pr["v"] + R.sum(0)
        self.U, self.logW = U, logW

    def _expect(self, X):
        d, k = self.mean.shape
        Eq = np.zeros((X.shape[1], k))
        for i in range(k):
            q = np.linalg.solve(self.U[:, :, i].T, X - self.mean[:, [i]])
            Eq[:, i] = d / self.kappa[i] + self.v[i] * (q * q).sum(0)
        ElogL = digamma(0.5 * (self.v[None, :] + 1 - np.arange(1, d + 1)[:, None])).sum(0) + d * np.log(2) + self.logW
        Elogpi = digamma(self.alpha) - digamma(self.alpha.sum())
        logRho = -0.5 * (Eq - (ElogL - d * np.log(2 * np.pi))) + Elogpi
        y = logRho.max(1, keepdims=True)
        lse = y + np.log(np.exp(logRho - y).sum(1, keepdims=True))
        lse[np.isinf(y)] = y[np.isinf(y)]
        self.logR = logRho - lse
        self.R = np.exp(self.logR)
        self.weights = self.R.sum(0) / self.R.shape[0]

    def _bound(self, X):
        pr = self.prior
        d, k = X.shape[0], self.R.shape[1]

        def lmg(x):
            x = np.atleast_1d(x)
            return d * (d - 1) / 4 * np.log(np.pi) + gammaln(x[None, :] + (1 - np.arange(1, d + 1)[:, None]) / 2).sum(0)

        Eq_z = float(np.sum(self.R * self.logR))
        Ep_pi = gammaln(k * pr["alpha"]) - k * gammaln(pr["alpha"])
        Eq_pi = gammaln(self.alpha.sum()) - gammaln(self.alpha).sum()
        Ep_mu = 0.5 * d * k * np.log(pr["kappa"])
        Eq_mu = 0.5 * d * np.log(self.kappa).sum()
        Ep_L = k * (-0.5 * pr["v"] * (pr["logW"] + d * np.log(2)) - lmg(0.5 * pr["v"])[0])
        Eq_L = float(np.sum(-0.5 * self.v * (self.logW + d * np.log(2)) - lmg(0.5 * self.v)))
        Ep_X = -0.5 * d * X.shape[1] * np.log(2 * np.pi)
        return 0 - Eq_z + Ep_pi - Eq_pi + Ep_mu - Eq_mu + Ep_L - Eq_L + Ep_X

    def sample(self, n, rng):
        w = self.weights if hasattr(self, "weights") else self.R.sum(0) / len(self.R)
        p = np.cumsum(w)
        z = np.searchsorted(p / p[-1], rng.random(n), side="right")
        d = self.mean.shape[0]
        X = np.zeros((n, d))
        for i in range(self.mean.shape[1]):
            s = z == i
            if s.any():
                S = self.U[:, :, i]
                try:
                    R = np.linalg.cholesky(S).T
                except np.linalg.LinAlgError:
                    R = S
                X[s] = (R.T @ rng.standard_normal((d, int(s.sum()))) + self.mean[:, [i]]).T
        return X


class OSPNSDE(LoopAlgorithm):
    """NSDE (DE/rand/1 + non-dominated sorting with distance truncation). When the hypervolume of the first front has grown
    by ``lam`` since the last prediction (and the front is not nearly full), or at generation 10, the movement of the front
    since then is modelled by per-objective AR(1) models, the front is forecast ``p`` generations ahead, every front member is
    locally optimised (SLSQP, evaluations charged) towards its forecast point, and the rest of the population is sampled
    from a variational Gaussian mixture of the optimised solutions; ``p`` is then halved."""

    def __init__(self, pop_size: int = 100, lam: float = 0.2, p: int = 50, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.lam, self.p = float(lam), int(p)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        F, C = objs(infills), cons(infills)
        self.front, _ = nd_sort(F, C if C.shape[1] else None, self.N)
        self.nadir = F.max(0)
        self.t_init, self.t = 2, 0
        self.PX, self.PF, self.PR, self.hyp = [], [], [], []
        self._set_optimum()

    def _select(self, pop):
        F, C, N = objs(pop), cons(pop), self.N
        front, maxf = nd_sort(F, C if C.shape[1] else None, N)
        nxt = front < maxf
        last = np.where(front == maxf)[0]
        nxt[last] = True
        Dm = np.sqrt(((F[last][:, None] - F[last][None]) ** 2).sum(-1))
        np.fill_diagonal(Dm, np.inf)
        nxt[last[truncate_lexi(Dm, int(nxt.sum()) - N)]] = False
        return pop[nxt], front[nxt]

    def _de_step(self):
        rng, N = self.rng, self.N
        X = decs(self.pop)
        mp = np.concatenate([np.arange(N), rng.integers(0, N, 2 * N)])
        off = self.evaluate(de(self.problem, X[mp[:N]], X[mp[N:2 * N]], X[mp[2 * N:]], rng=rng))
        self.pop, self.front = self._select(Population.merge(self.pop, off))

    def _local_opt(self, x0, target):
        from scipy.optimize import minimize
        cache = {}

        def fun(x):
            key = x.tobytes()
            if key not in cache:
                cache[key] = float(np.linalg.norm(objs(self.evaluate(np.clip(x, self.lower, self.upper)[None]))[0] - target))
            return cache[key]

        res = minimize(fun, x0, method="SLSQP", bounds=list(zip(self.lower, self.upper)),
                       options={"maxiter": 400, "ftol": 1e-6})
        return np.clip(res.x, self.lower, self.upper), res.fun

    def _osp(self):
        rng, N, M = self.rng, self.N, self.M
        C = self.t - 1
        S = np.where(self.PR[C] == 1)[0]
        order = np.argsort(self.PF[C][S, 0], kind="stable")
        phiS, indS = self.PF[C][S][order], self.PX[C][S][order]
        data = []
        for g in range(self.t_init - 1, self.t):
            i1 = np.where(self.PR[g] == 1)[0]
            Fg = self.PF[g][i1]
            if len(i1) != len(S):
                idx = np.argmin(((phiS[:, None] - Fg[None]) ** 2).sum(-1), 1)
            else:
                idx = np.argsort(Fg[:, 0], kind="stable")
            data.append(Fg[idx])
        data = np.stack(data, axis=2)                       # |S| x M x gens
        dist = np.sqrt(((data[:, :, 0] - data[:, :, -1]) ** 2).sum(1))
        dif = np.where(((data[:, :, -1] - data[:, :, 0]) < 0).sum(1) == M)[0]
        series = [data[i].T for i in range(len(data))]      # each gens x M
        if len(dif):
            b = dif[int(np.argmax(dist[dif]))]
            y2 = series[b][np.all(~np.isnan(series[b]), 1)]
            _, first = np.unique(y2[:, 0], return_index=True)
            y = y2[np.sort(first)]
            if len(y) < 3:
                y = y2
            target = ar1_forecast(y, series, self.p)
        else:
            target = ar1_forecast(None, series, self.p)
        XF = []
        for i in range(len(target)):
            x0, f0 = indS[i], phiS[i]
            base = self.evaluate(x0[None])
            d0 = float(np.linalg.norm(f0 - target[i]))
            x, dx = self._local_opt(x0, target[i])
            XF.append(self.evaluate(x[None]) if dx < d0 else base)
        XF = Population.merge(*XF)
        k = N - len(XF)
        if k:
            gm = VBGMM(decs(XF), len(XF), rng)
            Y = np.clip(gm.sample(k, rng), self.lower, self.upper)
            P = Population.merge(XF, self.evaluate(Y))
        else:
            P = XF
        F, Cc = objs(P), cons(P)
        self.pop = P
        self.front, _ = nd_sort(F, Cc if Cc.shape[1] else None, N)

    def step(self):
        N = self.N
        self.t += 1
        t = self.t
        F = objs(self.pop)
        self.PX.append(decs(self.pop)); self.PF.append(F); self.PR.append(self.front.copy())
        self.hyp.append(_hv(F[self.front == 1], self.nadir))
        if t == self.t_init:
            self.nadir = F.max(0)
        if t - self.t_init >= 3:
            alpha = self.hyp[self.t_init - 1] * (1 + self.lam)
            n1 = max(int((self.front == 1).sum()), self.M)        # MATLAB length() of an n x M matrix
            if t == 10 or (self.hyp[-1] >= alpha and n1 < 0.9 * N):
                self._osp()
                self.t_init = t + 1
                self.p = max(1, int(round(self.p / 2)))
                return
        self._de_step()
