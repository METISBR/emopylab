# emopylab 2026
"""MOEA-NZD (multi-objective evolutionary algorithm with nonzero detection).

Reference:
X. Wang, R. Cheng, and Y. Jin. Sparse large-scale multiobjective optimization by identifying nonzero
decision variables. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(10):
6280-6292.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, cosine_distance, decs, ga, nd_sort, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MOEANZD': {'constrained', 'large', 'many', 'multi', 'real', 'sparse'}}


def _prctile(x, p):
    """Percentile with the convention of the reference environment (positions ``p*n/100 + 0.5``)."""
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return np.nan
    pos = p / 100 * n + 0.5
    if pos <= 1:
        return x[0]
    if pos >= n:
        return x[-1]
    lo = int(np.floor(pos))
    return x[lo - 1] + (pos - lo) * (x[lo] - x[lo - 1])


def _quartiles(P):
    out = np.zeros((3, P.shape[1]))
    for i in range(P.shape[1]):
        col = P[P[:, i] != 0, i]
        out[:, i] = [_prctile(col, 75), _prctile(col, 50), _prctile(col, 25)]
    return np.nan_to_num(out, nan=0.0)


def _exact_kmeans(F):
    fmax, fmin = F.max(axis=0), F.min(axis=0)
    init = np.array([[fmax[0], fmax[1]], [fmax[0], fmin[1]], [fmin[0], fmax[1]], [fmin[0], fmin[1]]])
    C = init.copy()
    idx = np.zeros(len(F), dtype=int)
    for _ in range(100):
        with np.errstate(all="ignore"):
            d = np.sqrt(((F[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
        idx = np.argmin(np.where(np.isnan(d), np.inf, d), axis=1)
        for k in range(4):
            m = idx == k
            C[k] = F[m].mean(axis=0) if m.any() else init[k]
    return idx, C


class MOEANZD(LoopAlgorithm):
    """Sparse many-objective EA that starts from the all-zero decision vector: decision variables that are
    active in the mating pool are re-sampled inside the non-zero interquartile range, and in the last 30% of the
    run whole groups of zero variables are clamped, using a 4-cluster analysis of (magnitude, density)."""

    def initial_size(self):
        self.Z, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def _initialize_infill(self):
        self.Z, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.evaluate(np.zeros((self.pop_size, self.D)))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.start()
        self._set_optimum()

    def start(self):
        D = self.D
        self.zmin = self._feas_min(self.pop)
        self.Re0, self.Re1 = np.zeros(D, bool), np.ones(D, bool)
        self.t, self.t1, self.evaluation = 1, 1, 0
        self.demarcation = int(np.floor(self.max_FE * 0.7 / self.N))
        self.interval = max(int(np.floor(self.max_FE / self.N / 20)), 1)
        self.non0 = None

    @staticmethod
    def _feas_min(pop):
        C = cons(pop)
        feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
        return objs(pop)[feas].min(axis=0) if feas.any() else None

    def _dim_jud(self, P):
        rng = self.rng
        lo, up = self.lower, self.upper
        n, D = P.shape
        density = np.mean(P != 0, axis=0)
        q75, q50, q25 = _quartiles(P)
        km = np.zeros(D)
        ui, li = q50 > 0, q50 < 0
        km[ui] = q75[ui] / up[ui]
        km[li] = q25[li] / lo[li]
        with np.errstate(all="ignore"):
            km = np.sqrt(1 - (km - 1) ** 2)
        idx, Cp = _exact_kmeans(np.column_stack([km, density]))
        s = Cp[:, 0] + Cp[:, 1]
        a = int(np.argmax(np.where(np.isnan(s), -np.inf, s)))
        act = idx == a
        now = np.zeros(D, bool)
        re = self.Re0.copy()
        now[(ui.astype(int) + act) == 2] = True
        now[re] = False
        P = P.copy()
        if now.any():
            P[:, now] = rng.random((n, int(now.sum()))) * (up[now] - q75[now]) + q75[now]
        re[now] = True
        now[(li.astype(int) + act) == 2] = True
        now[re] = False
        if now.any():
            P[:, now] = -1.0 * rng.random((n, int(now.sum()))) * (q25[now] - lo[now]) + q25[now]
        re[now] = True
        return re, P

    def _dim_jud0(self, P):
        lo, up = self.lower, self.upper
        D = P.shape[1]
        density = np.mean(P != 0, axis=0)
        q75, q50, q25 = _quartiles(P)
        pn = np.zeros(D)
        ui, li = q50 > 0, q50 < 0
        pn[ui] = q75[ui] / up[ui]
        pn[li] = q25[li] / lo[li]
        with np.errstate(all="ignore"):
            pn = np.sqrt(1 - (pn - 1) ** 2)
        idx, Cp = _exact_kmeans(np.column_stack([pn, density]))
        order = np.argsort(np.where(np.isnan(Cp[:, 0]), np.inf, Cp[:, 0]), kind="stable")
        is0 = ((idx == order[0]).astype(int) + (idx == order[1]).astype(int)) == 1
        P = P.copy()
        P[:, is0] = 0
        return P, (~is0)

    def _ga0(self, P, non0):
        """Real/integer variables: SBX and polynomial mutation restricted to the currently non-zero variables."""
        rng, pr = self.rng, self.problem
        enc = self.encoding
        h = len(P) // 2
        P1, P2 = P[:h], P[h:2 * h]
        cols = np.where(np.isin(enc, (1, 2)))[0]
        off = ga(pr, P, [1, 20, 1, 20], rng=rng)                    # other variable types keep the standard operator
        if len(cols):
            a, b = P1[:, cols], P2[:, cols]
            n, d = a.shape
            mu = rng.random((n, d))
            beta = np.zeros((n, d))
            beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / 21)
            beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / 21)
            beta = beta * (-1.0) ** rng.integers(0, 2, (n, d))
            beta[rng.random((n, d)) < 0.5] = 1
            beta[np.repeat(rng.random((n, 1)) > 1, d, axis=1)] = 1
            o = np.vstack([(a + b) / 2 + beta * (a - b) / 2, (a + b) / 2 - beta * (a - b) / 2])
            lo, up = np.tile(self.lower[cols], (2 * n, 1)), np.tile(self.upper[cols], (2 * n, 1))
            o = np.minimum(np.maximum(o, lo), up)
            nz = non0[cols]
            site = rng.random((2 * n, d)) < (len(cols) / len(enc)) / max(int(non0.sum()), 1)
            mu = rng.random((2 * n, d))
            span = up - lo
            with np.errstate(all="ignore"):
                t = nz & site & (mu <= 0.5)
                o[t] = o[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (o[t] - lo[t]) / span[t]) ** 21) ** (1 / 21) - 1)
                t = nz & site & (mu > 0.5)
                o[t] = o[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - o[t]) / span[t]) ** 21) ** (1 / 21))
            off[:, cols] = o
        return off

    def _last_selection(self, F1, F2, K, Z, zmin):
        rng = self.rng
        F = np.vstack([F1, F2]) - zmin
        N, M = F.shape
        N1, N2, NZ = len(F1), len(F2), len(Z)
        w = np.zeros((M, M)) + 1e-6 + np.eye(M)
        ext = np.array([int(np.argmin(np.max(F / w[i], axis=1))) for i in range(M)])
        try:
            a = 1.0 / np.linalg.solve(F[ext], np.ones(M))
        except np.linalg.LinAlgError:
            a = np.full(M, np.nan)
        if np.any(np.isnan(a)):
            a = F.max(axis=0)
        F = F / a
        cosine = 1 - cosine_distance(F, Z)
        dist = np.linalg.norm(F, axis=1)[:, None] * np.sqrt(np.maximum(0, 1 - cosine ** 2))
        pi = np.argmin(dist, axis=1)
        d = dist[np.arange(N), pi]
        rho = np.bincount(pi[:N1], minlength=NZ)
        choose = np.zeros(N2, bool)
        zc = np.ones(NZ, bool)
        while choose.sum() < K:
            tmp = np.where(zc)[0]
            jm = np.where(rho[tmp] == rho[tmp].min())[0]
            j = tmp[jm[rng.integers(len(jm))]]
            I = np.where(~choose & (pi[N1:] == j))[0]
            if len(I):
                s = int(np.argmin(d[N1 + I])) if rho[j] == 0 else int(rng.integers(len(I)))
                choose[I[s]] = True
                rho[j] += 1
            else:
                zc[j] = False
        return choose

    def _select(self, pop):
        N = self.N
        zmin = np.ones(self.Z.shape[1]) if self.zmin is None else self.zmin
        C = cons(pop)
        front, maxf = nd_sort(objs(pop), C if C.size else None, N)
        nxt = front < maxf
        last = np.where(front == maxf)[0]
        ch = self._last_selection(objs(pop)[nxt], objs(pop)[last], N - int(nxt.sum()), self.Z, zmin)
        nxt[last[ch]] = True
        return pop[nxt]

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        self.evaluation += 1
        C = cons(self.pop)
        cvs = np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(self.pop))
        X = decs(self.pop[tournament(2, N, cvs, rng=rng)])
        next_step = np.mean(np.abs(self.Re1.astype(int) - self.Re0.astype(int))) >= 1.0 / self.D
        if self.evaluation < self.demarcation:
            if next_step and self.evaluation % self.interval == 0:
                if self.t != 1:
                    self.Re0 = self.Re1.copy()
                self.t += 1
                self.Re1, X = self._dim_jud(X)
            off = self.evaluate(ga(pr, X, [1, 20, 1, 1], rng=rng))
        else:
            if self.t1 == 1 or self.evaluation % self.interval == 0:
                Xo, self.non0 = self._dim_jud0(X)
                self.t1 += 1
            else:
                Xo = self._ga0(X, self.non0)
            off = self.evaluate(Xo)
        fm = self._feas_min(off)
        if fm is not None:
            self.zmin = fm if self.zmin is None else np.minimum(self.zmin, fm)
        self.pop = self._select(Population.merge(self.pop, off))
