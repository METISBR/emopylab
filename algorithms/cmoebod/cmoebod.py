# emopylab 2026
"""CMOEBOD (constrained multiobjective evolutionary Bayesian optimization based on decomposition).

Reference:
Z. Zhang, Y. Wang, G.Sun, and T. Pang. A novel evolutionary Bayesian optimization algorithm based on
decomposition for expensive constrained multiobjective optimization problems. IEEE Transactions on
Systems, Man, and Cybernetics: Systems, 2025.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, objs, uniform_point
from algorithms.community_utils.dace import DaceModel, norm_cdf, norm_pdf
from algorithms.community_utils.disk import _angles
from core.population import Population

ALGORITHM_FLAGS = {'CMOEBOD': {'constrained', 'expensive', 'integer', 'many', 'multi', 'real'}}


def _pof(C, CM):
    with np.errstate(all="ignore"):
        return np.prod(norm_cdf(-C / np.sqrt(CM)), axis=1) if C.shape[1] else np.ones(len(C))


def _gp_max(mu, s2):
    """Moments of the maximum of two independent Gaussians (Clark's approximation)."""
    with np.errstate(all="ignore"):
        tao = np.sqrt(s2[:, 0] + s2[:, 1])
        a = (mu[:, 0] - mu[:, 1]) / tao
        y = mu[:, 0] * norm_cdf(a) + mu[:, 1] * norm_cdf(-a) + tao * norm_pdf(a)
        x = (mu[:, 0] ** 2 + s2[:, 0]) * norm_cdf(a) + (mu[:, 1] ** 2 + s2[:, 1]) * norm_cdf(-a) + mu.sum(1) * tao * norm_pdf(a) - y ** 2
    return y, x


def cpob(F, FM, C, CM, lam, rng):
    """Index of the solution dominated (in probability, weighted by feasibility) by the fewest others under the Gaussian
    approximation of the augmented Tchebycheff value along ``lam``."""
    pof = _pof(C, CM)
    N, M = F.shape
    u, s2 = lam * F, np.abs(lam ** 2 * FM)
    y, x = _gp_max(u[:, :2], np.abs(s2[:, :2]))
    for i in range(2, M):
        y, x = _gp_max(np.column_stack([y, u[:, i]]), np.abs(np.column_stack([x, s2[:, i]])))
    y = y + 0.05 * (lam * F).sum(1)
    x = x + 0.05 ** 2 * (lam ** 2 * FM).sum(1)
    i_idx, j_idx = np.triu_indices(N, 1)
    mean = y[i_idx] - y[j_idx]
    mean = np.where(mean == 0, 1.0, mean)
    with np.errstate(all="ignore"):
        xp = norm_cdf(-mean / np.abs(np.sqrt(x[i_idx] + x[j_idx])))
    yp = 1 - xp
    xp, yp = -xp * pof[i_idx], -yp * pof[j_idx]
    a = (xp <= yp) & (xp != yp)
    b = ~a & (xp >= yp) & (xp != yp)
    dom = np.zeros((N, N), bool)
    dom[i_idx[a], j_idx[a]] = True
    dom[j_idx[b], i_idx[b]] = True
    cnt = dom.sum(0)
    best = np.where(cnt == cnt.min())[0]
    return int(best[rng.integers(0, len(best))]) if len(best) > 1 else int(best[0])


def _assign(Fn, V, pick):
    """Iterative association: every remaining vector takes one of the remaining solutions closest to it (``pick`` breaks ties
    among several); repeated until every vector is served."""
    ang = _angles(Fn, V)
    P, Vi = np.ones(len(Fn), bool), np.ones(len(V), bool)
    while Vi.any() and P.any():
        pe, ve = np.where(P)[0], np.where(Vi)[0]
        assoc = np.argmin(ang[np.ix_(pe, ve)], axis=1)
        for i in np.unique(assoc):
            cur = np.where(assoc == i)[0]
            best = 0 if len(cur) == 1 else pick(pe[cur], ve[i])
            P[pe[cur[best]]] = False
            Vi[ve[i]] = False
    return ~P


class CMOEBOD(LoopAlgorithm):
    """Kriging models of objectives and constraints; the surrogate GA keeps, for every reference vector, the solution most
    likely (in feasibility-weighted probabilistic dominance of the approximate Tchebycheff value) to be best. Infill points are
    drawn among the candidates farthest in angle from the evaluated set, by their average probability of dominating it."""

    def __init__(self, pop_size: int = 100, wmax: int = 20, mu: int = 5, beta: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax, self.mu, self.beta = int(wmax), int(mu), float(beta)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.V, self.pop_size = uniform_point(self.pop_size, self.M)
        self.NI = self.N
        P, _ = UniformPoint(self.NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop, self.P = infills, infills
        self.th_obj = 5.0 * np.ones((self.M, self.D))
        self.th_con = 5.0 * np.ones((cons(infills).shape[1], self.D))
        self._set_optimum()

    def _fit(self, X, y, th):
        D = self.D
        d1 = np.unique(X, axis=0, return_index=True)[1]
        d2 = np.unique(y, return_index=True)[1]
        dist = np.intersect1d(d1, d2)
        if len(dist) <= D + 1:
            dist = d1      # deviation (see community_utils/pea.py)
        m = DaceModel(X[dist], y[dist], "regpoly1", th, 1e-5 * np.ones(D), 100 * np.ones(D))
        return m, m.theta

    def _predict(self, X):
        po = [m.predict(X, mse=True) for m in self.mo]
        pc = [m.predict(X, mse=True) for m in self.mc]
        F, FM = np.column_stack([p[0] for p in po]), np.abs(np.column_stack([p[1] for p in po]))
        if pc:
            return F, FM, np.column_stack([p[0] for p in pc]), np.abs(np.column_stack([p[1] for p in pc]))
        return F, FM, np.zeros((len(X), 0)), np.zeros((len(X), 0))

    def step(self):
        rng = self.rng
        X, F, C = decs(self.pop), objs(self.pop), cons(self.pop)
        self.mo, self.mc = [], []
        for i in range(self.M):
            m, self.th_obj[i] = self._fit(X, F[:, i], self.th_obj[i])
            self.mo.append(m)
        for i in range(C.shape[1]):
            m, self.th_con[i] = self._fit(X, C[:, i], self.th_con[i])
            self.mc.append(m)
        # surrogate optimisation
        PX = decs(self.P)
        for _ in range(self.wmax):
            PX = np.vstack([PX, ga(self.problem, PX, rng=rng)])
            PF, PM, PC, PCM = self._predict(PX)
            lo, r = PF.min(0), np.maximum(PF.max(0) - PF.min(0), 10e-10)
            Fn, Mn = (PF - lo) / r, PM / r ** 2
            keep = _assign(Fn, self.V, lambda idx, v: cpob(Fn[idx], Mn[idx], PC[idx], PCM[idx], self.V[v], rng))
            PX, PF, PM, PC, PCM = PX[keep], PF[keep], PM[keep], PC[keep], PCM[keep]
        self._new_select(PX, PF, PM, PC, PCM)
        # population update
        F, C = objs(self.pop), cons(self.pop)
        Fn = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 10e-10)

        def scalar(idx, v):
            g = np.max(Fn[idx] * self.V[v], 1)
            cv = np.maximum(0, C[idx]).sum(1)
            best = np.where(cv == cv.min())[0]
            return int(best[np.argmin(g[best])]) if len(best) > 1 else int(best[0])

        nxt = _assign(Fn, self.V, scalar)
        if self.NI > len(self.V):
            pe = np.where(~nxt)[0]
            nxt[pe[rng.permutation(len(pe))[: self.NI - len(self.V)]]] = True
        self.P = self.pop[nxt]

    def _new_select(self, PX, PF, PM, PC, PCM):
        db = decs(self.pop)
        idx = [i for i in range(len(PX)) if np.sqrt(((db - PX[i]) ** 2).sum(1)).min() > 1e-5]
        if not idx:
            return
        if len(idx) <= self.mu:
            self.pop = Population.merge(self.pop, self.evaluate(PX[idx]))
            return
        PX, PF, PM, PC, PCM = PX[idx], PF[idx], PM[idx], PC[idx], PCM[idx]
        A2 = np.unique(objs(self.pop), axis=0)
        allf = np.vstack([A2, PF])
        zmin, r = allf.min(0), np.maximum(allf.max(0) - allf.min(0), 10e-10)
        A2, PF, PM = (A2 - zmin) / r, (PF - zmin) / r, PM / r ** 2
        alive = np.ones(len(PX), bool)
        while (~alive).sum() < self.mu:
            pe = np.where(alive)[0]
            ang = np.sort(_angles(PF[alive], A2), 1)[:, 0]
            order = np.argsort(-ang, kind="stable")[: min(int(np.ceil(self.beta * self.NI)), len(pe))]
            L = pe[order]
            pof = _pof(PC[L], PCM[L])
            with np.errstate(all="ignore"):
                pro = np.array([-np.prod(norm_cdf(-(PF[l] - A2) / np.sqrt(PM[l])), 1).sum() / len(A2) * pof[k] for k, l in enumerate(L)])
            k = L[int(np.argsort(pro, kind="stable")[0])]
            self.pop = Population.merge(self.pop, self.evaluate(PX[[k]]))
            A2 = (objs(self.pop) - zmin) / r
            alive[k] = False
