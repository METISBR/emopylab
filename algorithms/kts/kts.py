# emopylab 2026
"""KTS (kriging-assisted evolutionary algorithm with two search modes).

Reference:
Z. Song, H. Wang, B. Xue, M. Zhang, and Y. Jin. Balancing objective optimization and constraint
satisfaction in expensive constrained evolutionary multi-objective optimization. IEEE Transactions
on Evolutionary Computation, 2024, 28(5): 1286-1300.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import spea
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, kmeans, nd_sort, objs, tournament
from algorithms.community_utils.dace import DaceModel
from algorithms.community_utils.kta import adaptive_sampling, ibea_keep, lp_greedy
from algorithms.kta2.kta2 import update_ca, update_da
from core.population import Population

ALGORITHM_FLAGS = {'KTS': {'constrained', 'expensive', 'integer', 'multi', 'real'}}


def _spea_keep(F, fit, N):
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[spea.truncation(F[idx], int(nxt.sum()) - N)]] = False
    return nxt


def update_p(P, N, with_cons):
    F = objs(P)
    fit = spea.cal_fitness(F, cons(P) if with_cons and cons(P).shape[1] else None)
    nxt = np.where(_spea_keep(F, fit, N))[0]
    return P[nxt[np.argsort(fit[nxt], kind="stable")]]


def cal_q(F):
    N = len(F)
    with np.errstate(all="ignore"):
        Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
    I = np.max(Fn[:, None, :] - Fn[None, :, :], axis=2)
    C = np.abs(I).max(0)
    with np.errstate(all="ignore"):
        return 1.0 / ((-np.exp(-I / C[None, :] / 0.05)).sum(0) + 1)


class _Arch:
    """Predicted archive (decisions, objectives, constraints, prediction variances)."""

    def __init__(self, X, F, C, V):
        self.X, self.F, self.C, self.V = X, F, C, V

    @classmethod
    def of(cls, pop, M):
        C = cons(pop)
        return cls(decs(pop), objs(pop), C, np.zeros((len(pop), M + C.shape[1])))

    def cat(self, o):
        return _Arch(np.vstack([self.X, o.X]), np.vstack([self.F, o.F]), np.vstack([self.C, o.C]), np.vstack([self.V, o.V]))

    def take(self, idx):
        return _Arch(self.X[idx], self.F[idx], self.C[idx], self.V[idx])


class KTS(LoopAlgorithm):
    """Kriging-assisted search that switches between an unconstrained mode (KTA2-style two archives with adaptive sampling)
    and a constrained mode (SPEA2 with constrained dominance on surrogate objectives and constraints, then k-means based
    infill near the feasible non-dominated set). The mode follows the correlation between the convergence contribution and
    the constraint violation of the least-contributing evaluated solutions."""

    def __init__(self, pop_size: int = 100, tau: float = 0.6, phi: float = 0.2, mu: int = 20, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.tau, self.phi, self.mu = float(tau), -float(phi), int(mu)
        self.phi1, self.wmax1, self.mu1 = 0.1, 10, 5

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        P, _ = UniformPoint(self.N, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        rng = self.rng
        self.p = 1.0 / self.M
        self.pop = infills                                   # A1
        self.CA = update_ca(None, infills, self.N)
        self.DA1 = update_da(infills[:0], infills, self.N, self.p, rng) if len(infills) else infills
        self.P1 = self.P2 = infills
        self.nc = cons(infills).shape[1]
        self.theta = 5.0 * np.ones((self.M + self.nc, self.D))
        self.models = [None] * (self.M + self.nc)
        X = np.vstack([decs(self.DA1), decs(self.CA), decs(self.P2)])
        C = np.vstack([cons(self.DA1), cons(self.CA), cons(self.P2)])
        u = np.unique(np.round(X * 1e4) / 1e4, axis=0, return_index=True)[1]
        self._fit_cons(X[u], C[u])
        self._set_optimum()

    def _dace(self, X, y, i):
        D = self.D
        try:
            m = DaceModel(X, y, "regpoly0", self.theta[i], 1e-5 * np.ones(D), 100 * np.ones(D))
            self.theta[i] = m.theta
        except Exception:  # noqa: BLE001  - a single design site (see the constrained-mode rounding) cannot be fitted
            m = _Const(float(np.mean(y)))
        self.models[i] = m

    def _fit_cons(self, X, C):
        for j in range(self.nc):
            self._dace(X, C[:, j], self.M + j)

    def _predict(self, X):
        out, var = np.zeros((len(X), self.M + self.nc)), np.zeros((len(X), self.M + self.nc))
        for j, m in enumerate(self.models):
            out[:, j], var[:, j] = m.predict(X, mse=True)
        return _Arch(X, out[:, : self.M], out[:, self.M:], var)

    def step(self):
        rng, M, N = self.rng, self.M, self.N
        A1 = self.pop
        q = cal_q(objs(A1))
        o = np.argsort(-q, kind="stable")
        q = q[o]
        cv = np.maximum(cons(A1), 0).sum(1)[o] if self.nc else np.zeros(len(q))
        with np.errstate(all="ignore"):
            r = np.corrcoef(q[len(q) - self.mu - 1:], cv[len(cv) - self.mu - 1:])[0, 1]
        if r < self.phi:
            mode = 1
        elif r < self.tau:
            mode = 0 if rng.random() < 0.5 else 1
        else:
            mode = 0
        DA = self.DA1 if mode == 0 else self.P1
        X, F = decs(A1), objs(A1)
        for i in range(M):
            self._dace(X, F[:, i], i)
        if mode == 1 and self.nc:
            Xc = np.vstack([decs(DA), decs(self.CA), decs(self.P2)])
            Cc = np.vstack([cons(DA), cons(self.CA), cons(self.P2)])
            u = np.unique(np.round(Xc, -4), axis=0, return_index=True)[1]    # literal: rounding to the 10^4 place
            self._fit_cons(Xc[u], Cc[u])
        CCA, CP2, CDA = _Arch.of(self.CA, M), _Arch.of(self.P2, M), _Arch.of(DA, M)
        for _ in range(self.wmax1):
            if mode == 0:
                h = int(np.ceil(N / 2))
                a, b = rng.integers(0, len(CCA.F), h), rng.integers(0, len(CCA.F), h)
                dom = (CCA.F[a] < CCA.F[b]).any(1).astype(int) - (CCA.F[a] > CCA.F[b]).any(1).astype(int)
                pc = np.vstack([CCA.X[np.concatenate([a[dom == 1], b[dom != 1]])], CDA.X[rng.integers(0, len(CDA.X), h)]])
                pm = CCA.X[rng.integers(0, len(CCA.X), N)]
                off = np.vstack([ga(self.problem, pc, (1, 20, 0, 0), rng=rng), ga(self.problem, pm, (0, 0, 1, 20), rng=rng)])
            else:
                f1 = spea.cal_fitness(CP2.F, CP2.C if self.nc else None)
                f2 = spea.cal_fitness(CDA.F)
                off = np.vstack([ga(self.problem, CP2.X[tournament(2, N, f1, rng=rng)], rng=rng),
                                 ga(self.problem, CDA.X[tournament(2, N, f2, rng=rng)], rng=rng)])
            pop = self._predict(off)
            c = CCA.cat(pop)
            CCA = c.take(ibea_keep(c.F, N)) if len(c.F) > N else c
            d = CDA.cat(pop)
            if mode == 0:
                with np.errstate(all="ignore"):
                    pre = (d.F - d.F.min(0)) / (d.F.max(0) - d.F.min(0))
                nd, _ = nd_sort(d.F, None, 1)
                d, pre = d.take(np.where(nd == 1)[0]), pre[nd == 1]
                if len(d.F) > N:
                    ch = np.zeros(len(d.F), bool)
                    ch[rng.permutation(M)[0]] = True
                    ch, _ = lp_greedy(pre, ch, N, self.p)
                    d = d.take(np.where(ch)[0])
                CDA = d
            else:
                CDA = d.take(np.where(_spea_keep(d.F, spea.cal_fitness(d.F), N))[0])
            e = CP2.cat(pop)
            fe = spea.cal_fitness(e.F, e.C if self.nc else None)
            CP2 = e.take(np.where(_spea_keep(e.F, fe, N))[0])
        if mode == 0:
            ax = decs(A1)

            def notin(arc):
                idx = [i for i in range(len(arc.X)) if not np.any(np.all(ax == arc.X[i], 1))]
                idx = np.array(idx, dtype=int)
                if len(idx):
                    idx = idx[np.unique(arc.X[idx], axis=0, return_index=True)[1]]
                return arc.take(idx)

            CCA, CDA = notin(CCA), notin(CDA)
            if len(CCA.X) == 0 or len(CDA.X) == 0:
                cand = np.vstack([CCA.X, CDA.X])[: self.mu1]
            else:
                cand = adaptive_sampling(CCA.F, CDA.F, CCA.X, CDA.X, CDA.V, objs(DA), decs(DA), self.mu1, self.p, self.phi1, rng,
                                         small_shortcut=True)
        else:
            cand = self._kccmo(CP2)
        if len(cand) == 0:
            return
        cand = np.unique(cand, axis=0)
        off = self.evaluate(cand)
        for i in range(len(off)):
            if np.sqrt(((decs(self.pop) - decs(off)[i]) ** 2).sum(1)).min() > 1e-5:
                self.pop = Population.merge(self.pop, off[[i]])
        self.CA = update_ca(self.CA, off, N)
        self.DA1 = update_da(self.DA1, off, N, self.p, rng)
        self.P1 = update_p(Population.merge(self.P1, off), N, False)
        self.P2 = update_p(Population.merge(self.P2, off), N, True)

    def _kccmo(self, arc):
        rng = self.rng
        C2 = cons(self.P2)
        feas = np.all(C2 <= 0, 1) if C2.shape[1] else np.ones(len(self.P2), bool)
        ref = objs(self.P2)[feas]
        if len(ref):
            f, _ = nd_sort(ref, None, 1)
            ref = ref[f == 1]
        else:
            ref = objs(self.P2)
        F = arc.F
        cv = np.maximum(0, arc.C).sum(1) if arc.C.shape[1] else np.zeros(len(F))
        k = (F[:, None] < F[None]).any(-1).astype(int) - (F[:, None] > F[None]).any(-1).astype(int)
        dom = (cv[:, None] < cv[None]) | ((cv[:, None] == cv[None]) & (k == 1))
        S = dom.sum(1)
        R = S @ dom
        fit = R + 1.0 / (np.sqrt(((F[:, None] - ref[None]) ** 2).sum(-1)).min(1) + 2)
        lab = kmeans(F, self.mu1, rng)
        pick = [np.where(lab == c)[0][int(np.argmin(fit[lab == c]))] for c in np.unique(lab)]
        return arc.X[np.array(pick, dtype=int)]


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, X, mse=False):
        y = np.full(len(X), self.v)
        return (y, np.zeros(len(X))) if mse else y
