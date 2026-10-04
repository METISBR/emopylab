# emopylab 2026
"""ESB-CEO (bayesian co-evolutionary optimization based entropy search).

Reference:
H. Bian, J. Tian, J. Yu, and H. Yu. Bayesian co-evolutionary optimization based entropy search for
high-dimensional many-objective optimization. Knowledge-Based Systems, 2023, 274: 110630.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, kmeans, nd_sort, objs, uniform_point
from algorithms.community_utils.dace import DaceModel
from algorithms.k_rvea.k_rvea import _angles, _argmin_rows, _gamma
from core.population import Population

ALGORITHM_FLAGS = {'ESBCEO': {'expensive', 'integer', 'many', 'multi', 'real'}}


def fcm(X, c, rng, expo=2.0, max_iter=100, min_impro=0.05):
    """Fuzzy c-means (random fuzzy partition start); returns the cluster centres."""
    U = rng.random((c, len(X)))
    U = U / U.sum(0)
    prev = None
    center = None
    for _ in range(max_iter):
        mf = U ** expo
        center = mf @ X / mf.sum(1, keepdims=True)
        dist = np.sqrt(((X[None, :, :] - center[:, None, :]) ** 2).sum(-1))
        obj = float(((dist ** 2) * mf).sum())
        with np.errstate(all="ignore"):
            tmp = dist ** (-2 / (expo - 1))
            U = tmp / tmp.sum(0)
        U = np.nan_to_num(U, nan=1.0 / c)
        if prev is not None and abs(obj - prev) < min_impro:
            break
        prev = obj
    return center


def _entropy(DO, n_dec, m, k=10):
    """Shannon entropy of the (normalised) ``n_dec``-th leading column over the k nearest neighbours in the next ``m``
    columns (the reference fixes ``n_dec`` = 10)."""
    Y = DO[:, n_dec:n_dec + m]
    d = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))
    kk = min(k, len(DO))
    col = DO[:, min(n_dec, DO.shape[1]) - 1]
    H = np.zeros(len(DO))
    for i in range(len(DO)):
        knn = np.argsort(d[i], kind="stable")[:kk]
        v = col[knn]
        s = v.sum()
        P = np.full(kk, 1.0 / kk) if s == 0 else v / s
        with np.errstate(all="ignore"):
            h = np.where(P == 0, 0.0, P * np.log2(P))
        H[i] = -np.nansum(h)
    return H


class ESBCEO(LoopAlgorithm):
    """Fuzzy-c-means partitioned local Kriging models (each fitted on ``L1`` points of its cluster) drive a cooperative
    surrogate search that alternates MOEA/D-DE and RVEA populations, merged every third round by non-domination and a
    neighbourhood entropy; ``Ke`` k-means representatives are evaluated, chosen by a budget-scheduled trade-off between
    Lp-distance to the origin and entropy."""

    def __init__(self, pop_size: int = 100, Ke: int = 5, delta: float = 0.9, nr: int = 2, L1: int = 80, L2: int = 20,
                 sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Ke, self.delta, self.nr, self.L1, self.L2 = int(Ke), float(delta), int(nr), int(L1), int(L2)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        NI = int(np.floor(100 + self.D / 10))       # the reference uses the non-integer 100 + D/10
        P, _ = UniformPoint(NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.L1 = min(self.L1, len(infills))
        self._set_optimum()

    def _models(self):
        X, F = decs(self.pop), objs(self.pop)
        D, M = self.D, self.M
        c = 1 + int(np.ceil((len(X) - self.L1) / self.L2))
        centers = fcm(X, c, self.rng)
        dis = np.sqrt(((X[:, None] - centers[None]) ** 2).sum(-1))
        group = np.argsort(-dis, axis=0, kind="stable")[: self.L1]     # literal: the L1 farthest points of each centre
        models = []
        for i in range(c):
            g = group[:, i]
            row = []
            for j in range(M):
                row.append(DaceModel(X[g], F[g, j], "regpoly0", 5.0 * np.ones(D), 1e-5 * np.ones(D), 100 * np.ones(D)))
            models.append(row)
        return models, centers

    def _pred(self, X, models, centers):
        X = np.atleast_2d(X)
        idx = np.argmin(((X[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        out = np.zeros((len(X), self.M))
        for c in np.unique(idx):
            s = idx == c
            out[s] = np.column_stack([m.predict(X[s]) for m in models[c]])
        return out

    def _ga_bound(self, P):
        rng = self.rng
        n = len(P) // 2
        A, B = P[:n], P[n:2 * n]
        d = P.shape[1]
        mu = rng.random((n, d))
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
        beta = beta * (-1.0) ** rng.integers(0, 2, (n, d))
        beta[rng.random((n, d)) < 0.5] = 1
        off = (A + B) / 2 + beta * (A - B) / 2
        lo, up = np.broadcast_to(self.lower, off.shape), np.broadcast_to(self.upper, off.shape)
        s, m = rng.random((n, d)) < 1.0 / d, rng.random((n, d))
        off = np.minimum(np.maximum(off, lo), up)
        with np.errstate(all="ignore"):
            t = s & (m <= 0.5)
            off[t] = off[t] + (up[t] - lo[t]) * ((2 * m[t] + (1 - 2 * m[t]) * (1 - (off[t] - lo[t]) / (up[t] - lo[t])) ** 21) ** (1 / 21) - 1)
            t = s & (m > 0.5)
            off[t] = off[t] + (up[t] - lo[t]) * (1 - (2 * (1 - m[t]) + 2 * (m[t] - 0.5) * (1 - (up[t] - off[t]) / (up[t] - lo[t])) ** 21) ** (1 / 21))
        return off

    def step(self):
        rng, M = self.rng, self.M
        models, centers = self._models()
        AX, AF = decs(self.pop), objs(self.pop)
        Z = AF.min(0)
        W, N = uniform_point(self.N, M)
        T = int(np.ceil(N / 10))
        B = np.argsort(np.sqrt(((W[:, None] - W[None]) ** 2).sum(-1)), axis=1, kind="stable")[:, :T]
        dup = rng.integers(0, len(self.pop), N)
        PX, PF = AX[dup].copy(), AF[dup].copy()
        V0, N1 = uniform_point(len(self.pop), M)
        V = V0.copy()
        P2X, P2F = AX.copy(), AF.copy()
        for eva in range(1, 21):
            for i in range(N):
                P = B[i][rng.permutation(T)] if rng.random() < self.delta else rng.permutation(N)
                ox = de(self.problem, PX[[i]], PX[[P[0]]], PX[[P[1]]], rng=rng)
                of = self._pred(ox, models, centers)[0]
                Z = np.minimum(Z, of)
                g_old = np.max(np.abs(PF[P] - Z) * W[P], 1)
                g_new = np.max(np.abs(of - Z) * W[P], 1)
                hit = P[np.where(g_old > g_new)[0][: self.nr]]
                PX[hit], PF[hit] = ox[0], of
            X1, F1 = PX.copy(), PF.copy()
            mate = rng.integers(0, len(P2X), N1)
            od = self._ga_bound(P2X[mate])
            ofo = self._pred(od, models, centers)
            allX, allF = np.vstack([P2X, od]), np.vstack([P2F, ofo])
            theta = (self.FE / self.max_FE) ** 2
            Fs = P2F - P2F.min(0)                        # literal: selection reads only the previous population
            ang = _angles(Fs, V)
            assoc = _argmin_rows(ang)
            gam = _gamma(V)
            idx = []
            for k in np.unique(assoc):
                cur = np.where(assoc == k)[0]
                apd = (1 + M * theta * ang[cur, k] / gam[k]) * np.sqrt((Fs[cur] ** 2).sum(1))
                idx.append(cur[int(np.argmin(apd))])
            idx = np.array(idx, dtype=int)
            X2, F2 = allX[idx], allF[idx]
            if int(np.ceil(self.FE / N1)) % int(np.ceil(0.1 * self.max_FE / N1)) == 0:
                V[:N1] = V0 * (F2.max(0) - F2.min(0))
            if eva % 3 == 0:
                f1, _ = nd_sort(F1, None, np.inf)
                f2, _ = nd_sort(F2, None, np.inf)
                PX = np.vstack([X1[f1 == 1], X2[f2 == 1]])
                PF = np.vstack([F1[f1 == 1], F2[f2 == 1]])
                if len(PX) > N:
                    dist = ((np.abs(PF) ** (1.0 / M)).sum(1)) ** M
                    order = np.argsort(dist, kind="stable")
                    pos = [int(np.where(order == j)[0][0]) for j in range(N)]    # literal: rows whose sorted index equals j
                    PX, PF = PX[pos], PF[pos]
                else:
                    tag = 1
                    while len(PX) + (f1 == tag + 1).sum() + (f2 == tag + 1).sum() < N and tag < 10 ** 6:
                        tag += 1
                        PX = np.vstack([PX, X1[f1 == tag], X2[f2 == tag]])
                        PF = np.vstack([PF, F1[f1 == tag], F2[f2 == tag]])
                        if not ((f1 > tag) & np.isfinite(f1)).any() and not ((f2 > tag) & np.isfinite(f2)).any():
                            break
                    Dx = np.vstack([X1[f1 == tag + 1], X2[f2 == tag + 1]])
                    Do = np.vstack([F1[f1 == tag + 1], F2[f2 == tag + 1]])
                    if len(Dx) and len(PX) < N:
                        H = _entropy(np.hstack([Dx, Do]), 10, M)
                        o = np.argsort(-H, kind="stable")[: N - len(PX)]
                        PX, PF = np.vstack([PX, Dx[o]]), np.vstack([PF, Do[o]])
            else:
                PX, PF = X1, F1
            P2X, P2F = PX.copy(), PF.copy()
        lab = kmeans(PX, self.Ke, rng)
        q = -0.5 * np.cos(self.FE / self.max_FE * np.pi) + 0.5
        pick = []
        for c in np.unique(lab):
            ind = np.where(lab == c)[0]
            tx = PX[ind]
            to = self._pred(tx, models, centers)
            H = _entropy(np.hstack([tx, to]), 10, M)
            dist = ((np.abs(to) ** (1.0 / M)).sum(1)) ** M
            ei = (1 - q) * dist - q * H
            pick.append(ind[int(np.argmax(ei))])
        self.pop = Population.merge(self.pop, self.evaluate(PX[pick]))
