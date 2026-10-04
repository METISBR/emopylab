# emopylab 2026
"""HeE-MOEA (multiobjective evolutionary algorithm with heterogeneous ensemble based).

Reference:
D. Guo, Y. Jin, J. Ding, and T. Chai. Heterogeneous ensemble-based infill criterion for evolutionary
multiobjective optimization of expensive problems. IEEE Transactions on Cybernetics, 2019, 49(3):
1012-1025.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga, nd_sort, objs, tournament
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population
from util.svm import SVR

ALGORITHM_FLAGS = {'HeEMOEA': {'expensive', 'integer', 'multi', 'real'}}

_STR = [(a, b) for a in ("FE", "FS", "NONE") for b in ("RBF1", "SVM", "RBF2")]


def _pd(A, B):
    return np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0))


class NewRB:
    """Incrementally grown RBF network: neurons (spread ``spread``, i.e. bias 0.8326/spread) are added one at a time at
    the training input that most reduces the squared error of the least-squares linear output layer (with bias), until
    the mean squared error reaches ``goal`` or ``max_neurons`` neurons exist."""

    def __init__(self, goal, spread=1.0, max_neurons=None):
        self.goal, self.b, self.maxn = float(goal), 0.8326 / spread, max_neurons

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float).reshape(len(X), -1)
        n = len(X)
        A = np.exp(-(_pd(X, X) * self.b) ** 2)          # candidate responses (n x n)
        maxn = n if self.maxn is None else min(n, int(self.maxn))
        chosen = []
        H = np.ones((n, 1))
        W = np.linalg.lstsq(H, Y, rcond=None)[0]
        err = ((H @ W - Y) ** 2).mean()
        while err > self.goal and len(chosen) < maxn:
            best, best_e, best_W = None, np.inf, None
            for c in range(n):
                if c in chosen:
                    continue
                Hc = np.column_stack([A[:, chosen + [c]], np.ones(n)])
                Wc = np.linalg.lstsq(Hc, Y, rcond=None)[0]
                e = ((Hc @ Wc - Y) ** 2).mean()
                if e < best_e:
                    best, best_e, best_W = c, e, Wc
            if best is None:
                break
            chosen.append(best)
            err, W = best_e, best_W
        self.C = X[chosen]
        self.W = W
        return self

    def predict(self, Xq):
        Xq = np.atleast_2d(Xq)
        H = np.column_stack([np.exp(-(_pd(Xq, self.C) * self.b) ** 2), np.ones(len(Xq))]) if len(self.C) else np.ones((len(Xq), 1))
        return H @ self.W


class ClusterRBF:
    """RBF network with k-means centres (restarted until convergence within 100 iterations), spreads = distance to the
    nearest other centre, and a pseudo-inverse output layer with bias."""

    def __init__(self, k, rng):
        self.k, self.rng = int(k), rng

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        n = len(X)
        for _ in range(100):
            C = X[self.rng.integers(0, n, self.k)].copy()
            it = 1
            while it < 100:
                lab = np.argmin(_pd(X, C), 1)
                old = C.copy()
                for i in range(self.k):
                    if (lab == i).any():
                        C[i] = X[lab == i].mean(0)
                    else:
                        C[i] = np.nan                           # mean of an empty class (reference behaviour)
                if np.array_equal(C, old, equal_nan=True):
                    break
                it += 1
            if it < 100 and not np.isnan(C).any():
                break
        C = np.nan_to_num(C)
        Dc = _pd(C, C)
        np.fill_diagonal(Dc, Dc.max() + 1)
        self.S = np.maximum(Dc.min(0), 1e-12)
        self.C = C
        H = np.exp(-(_pd(C, X) / self.S[:, None]) ** 2)
        Hx = np.vstack([H, np.ones(n)])
        W = Y.T @ np.linalg.pinv(Hx)
        self.W2, self.B2 = W[:, : self.k], W[:, self.k]
        return self

    def predict(self, Xq):
        H = np.exp(-(_pd(self.C, np.atleast_2d(Xq)) / self.S[:, None]) ** 2)
        return (self.W2 @ H + self.B2[:, None]).T


def _discor(x, y=None):
    """Distance-correlation score as written in the reference: ``norm`` of a matrix is its 2-norm (a scalar), so the
    double-centred "distance matrices" are identically zero and the ratio is 0/0 = NaN (0 for a single variable)."""
    def dc(a, b):
        A = a - a - a + a            # scalar minus its row, column and grand means
        B = b - b - b + b
        with np.errstate(all="ignore"):
            return np.sqrt(A * B) / np.sqrt(np.sqrt(A * A) * np.sqrt(B * B))

    if y is not None:
        return float(dc(np.linalg.norm(x, 2), np.linalg.norm(np.atleast_2d(y).T, 2)))
    if x.shape[1] == 1:
        return 0.0
    return float(np.mean([dc(np.linalg.norm(x[:, [i]], 2), np.linalg.norm(np.delete(x, i, 1), 2)) for i in range(x.shape[1])]))


def pso_select(x, y, rng, m=20, iters=30, thr=0.5, c1=2.0, c2=2.0, w=1.05):
    """Binary PSO feature selection on the distance-correlation cost (NaN costs follow MATLAB's NaN-ignoring min)."""
    D = x.shape[1]
    vmax = np.full(D, 0.1)

    def cost(P):
        f = np.full(len(P), 100.0)
        for i, p in enumerate(P):
            idx = np.where(p == 1)[0]
            if len(idx):
                f[i] = -0.8 * _discor(x[:, idx], y) + 0.2 * _discor(x[:, idx])
        return f

    def nanargmin(f):
        return 0 if np.all(np.isnan(f)) else int(np.nanargmin(f))

    p = (rng.random((m, D)) >= thr).astype(float)
    v = vmax * rng.random((m, D))
    fit = cost(p)
    k = nanargmin(fit)
    zbest, fz = p[k].copy(), fit[k]
    gbest, fg = p.copy(), fit.copy()
    for _ in range(iters):
        v = w * v + c1 * rng.random((m, D)) * (gbest - p) + c2 * rng.random((m, D)) * (zbest - p)
        v = np.clip(v, -vmax, vmax)
        p = (p + v >= thr).astype(float)
        fit = cost(p)
        imp = fit < fg
        fg[imp], gbest[imp] = fit[imp], p[imp]
        k = nanargmin(fit)
        if fit[k] < fz:
            fz, zbest = fit[k], p[k].copy()
    return np.where(zbest == 1)[0]


def _pca95(X):
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lat = s ** 2
    cum = np.cumsum(lat) / lat.sum()
    k = int(np.searchsorted(cum, 0.95) + 1)
    return Vt[:k].T


class _MinMax01:
    def __init__(self, X):
        self.lo = X.min(0)
        g = X.max(0) - self.lo
        self.g = np.where(g > 0, g, 1.0)

    def apply(self, X):
        return (X - self.lo) / self.g

    def reverse(self, Y):
        return Y * self.g + self.lo


class HeEMOEA(LoopAlgorithm):
    """Nine heterogeneous surrogates (feature extraction by PCA, feature selection by binary PSO, or all variables x an
    incrementally grown RBF network, a sigmoid-kernel SVR or a clustered RBF network) are averaged; NSGA-II minimises the
    lower confidence bound mean - 2 std of the ensemble, and ``Ke`` k-means centres of the new non-dominated candidates are
    evaluated per iteration. Training data are capped at ``L`` points."""

    def __init__(self, pop_size: int = 100, Ke: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Ke = int(Ke)

    def _initialize_infill(self):
        self.NI = 11 * self.D - 1
        self.L = self.NI + 25
        return self.evaluate(self.lower + lhs_design(self.rng, self.NI, self.D) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        sel = pso_select(decs(infills), objs(infills)[:, 0], self.rng)
        self.sel = sel if len(sel) else np.arange(self.D)      # an empty selection would leave FS models without inputs
        self._set_optimum()

    def _train(self, X, Y):
        rng, M, D = self.rng, self.M, self.D
        Mp = _pca95(X)
        models = []
        for fs, reg in _STR:
            proj = (lambda Z, Mp=Mp: Z @ Mp) if fs == "FE" else (lambda Z: Z[:, self.sel]) if fs == "FS" else (lambda Z: Z)
            Xt = proj(X)
            if reg == "RBF1":
                goal = np.sqrt(((Y.max(0) - Y.min(0)) ** 2).sum()) * 0.05
                m = NewRB(goal, 1.0, Xt.shape[1]).fit(Xt, Y)
                pred = (lambda Z, m=m, proj=proj: m.predict(proj(Z)))
            elif reg == "SVM":
                ps, qs = _MinMax01(Xt), _MinMax01(Y)
                svs = [SVR(C=1.0, epsilon=0.1, standardize=False, kernel="sigmoid").fit(ps.apply(Xt), qs.apply(Y)[:, j]) for j in range(M)]
                pred = (lambda Z, ps=ps, qs=qs, svs=svs, proj=proj:
                        qs.reverse(np.column_stack([s.predict(ps.apply(proj(Z))) for s in svs])))
            else:
                m = ClusterRBF(int(round(np.sqrt(M + D) + 3)), rng).fit(Xt, Y)
                pred = (lambda Z, m=m, proj=proj: m.predict(proj(Z)))
            models.append(pred)
        return models

    @staticmethod
    def _lcb(models, X):
        P = np.stack([m(X) for m in models])
        return P.mean(0) - 2 * np.sqrt(P.var(0, ddof=1))

    def _nsga2(self, F, N):
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        cd = crowding(F, front)
        last = np.where(front == maxf)[0]
        nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
        return nxt, front, cd

    def _kmean(self, P, trX):
        rng, D, Ke = self.rng, self.D, self.Ke
        Q = []
        for x in P:
            pool = np.vstack([trX] + ([np.array(Q)] if Q else []))
            if not (np.sqrt(((pool - x) ** 2).sum(1)) <= 1e-6).any():
                Q.append(x)
        if len(Q) < Ke:
            return None, 100
        Q = np.array(Q)
        C = Q[rng.permutation(len(Q))[:Ke]].copy()
        n = 1
        while n < 100:
            lab = np.argmin(_pd(Q, C), 1)
            old = C.copy()
            for i in range(Ke):
                C[i] = Q[lab == i].mean(0) if (lab == i).any() else np.nan
            if np.array_equal(C, old, equal_nan=True):
                break
            n += 1
        return C, n

    def step(self):
        rng, N, D, L = self.rng, self.N, self.D, self.L
        X, Y = decs(self.pop), objs(self.pop)
        if len(X) > L:
            front, _ = nd_sort(Y, None, len(X))
            o = np.argsort(front, kind="stable")
            h = L // 2
            rest = o[h:][rng.permutation(len(o) - h)[: L - h]]
            X, Y = np.vstack([X[o[:h]], X[rest]]), np.vstack([Y[o[:h]], Y[rest]])
        models = self._train(X, Y)
        C, nn = None, 0
        while nn == 100 or nn == 0 or C is None or np.isnan(C).any():
            P = self.lower + rng.random((N, D)) * (self.upper - self.lower)
            PF = self._lcb(models, P)
            keep, front, cd = self._nsga2(PF, N)
            P, PF, front, cd = P[keep], PF[keep], front[keep], cd[keep]
            for _ in range(self.max_FE // N):
                mate = tournament(2, N, front, -cd, rng=rng)
                off = ga(self.problem, P[mate], (1, 20, 1, 20), rng=rng)
                allX, allF = np.vstack([P, off]), np.vstack([PF, self._lcb(models, off)])
                keep, front, cd = self._nsga2(allF, N)
                P, PF, front, cd = allX[keep], allF[keep], front[keep], cd[keep]
            C, nn = self._kmean(P, X)
        self.pop = Population.merge(self.pop, self.evaluate(C))
