# emopylab 2026
"""MOEA-D-EGO (mOEA/D with efficient global optimization).

Reference:
Q. Zhang, W. Liu, E. Tsang, and B. Virginas. Expensive multiobjective optimization by MOEA/D with
Gaussian process model. IEEE Transactions on Evolutionary Computation, 2010, 14(3): 456-474.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, kmeans, nd_sort, neighbors_of, objs, uniform_point
from algorithms.community_utils.dace import DaceModel, norm_cdf, norm_pdf
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population

ALGORITHM_FLAGS = {'MOEADEGO': {'expensive', 'integer', 'multi', 'real'}}

L1, L2 = 80, 20
NUM_WEIGHTS = [200, 210, 295, 456, 462]


def _pdist(A, B):
    return np.sqrt(np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0))


def _fcm(rng, data, k, expo=2.0, max_iter=100, min_impro=0.05):
    """Fuzzy c-means; returns the cluster centres."""
    n = len(data)
    U = rng.random((k, n))
    U /= U.sum(axis=0)
    prev = None
    for i in range(max_iter):
        mf = U ** expo
        center = (mf @ data) / mf.sum(axis=1, keepdims=True)
        dist = _pdist(center, data)
        obj = float(np.sum(dist ** 2 * mf))
        with np.errstate(all="ignore"):
            tmp = dist ** (-2 / (expo - 1))
            U = tmp / tmp.sum(axis=0)
        if prev is not None and abs(obj - prev) < min_impro:
            break
        prev = obj
    return center


def _gp_model_fcm(rng, X, Y):
    K, M = Y.shape
    D = X.shape[1]
    if K <= L1:
        centers = _fcm(rng, X, 1)
        theta = K ** (-1.0 / K) * np.ones(D)
        return centers, [[DaceModel(X, Y[:, j], "regpoly0", theta, 1e-6 * np.ones(D), 20 * np.ones(D)) for j in range(M)]]
    csize = 1 + int(np.ceil((K - L1) / L2))
    centers = _fcm(rng, X, csize)
    order = np.argsort(_pdist(X, centers), axis=0, kind="stable")
    theta = L1 ** (-1.0 / L1) * np.ones(D)
    models = []
    for i in range(csize):
        idx = order[:L1, i]
        models.append([DaceModel(X[idx], Y[idx, j], "regpoly0", theta, 1e-6 * np.ones(D), 20 * np.ones(D)) for j in range(M)])
    return centers, models


def _gp_eval(X, models, centers, want_std=True):
    X = np.atleast_2d(X)
    idx = np.argmin(_pdist(X, centers), axis=1)
    M = len(models[0])
    u = np.zeros((len(X), M))
    mse = np.zeros((len(X), M))
    for c in np.unique(idx):
        rows = np.where(idx == c)[0]
        for j in range(M):
            y, e = models[c][j].predict(X[rows], mse=True)
            u[rows, j], mse[rows, j] = y, e
    if not want_std:
        return u
    mse[mse < 0] = 0
    return u, np.sqrt(mse)


def _max_of_2(mu, sig2):
    tao = np.sqrt(np.sum(sig2, axis=1))
    with np.errstate(all="ignore"):
        alpha = (mu[:, 0] - mu[:, 1]) / tao
    y = mu[:, 0] * norm_cdf(alpha) + mu[:, 1] * norm_cdf(-alpha) + tao * norm_pdf(alpha)
    s2 = (mu[:, 0] ** 2 + sig2[:, 0]) * norm_cdf(alpha) + (mu[:, 1] ** 2 + sig2[:, 1]) * norm_cdf(-alpha) + np.sum(mu, axis=1) * tao * norm_pdf(alpha)
    s2 = s2 - y ** 2
    s2 = np.where(s2 < 0, 0.0, s2)
    return y, s2


def _get_eti(u, sigma, ref, gbest, z):
    g_mu = ref * (u - z)
    g_sig = ref * sigma
    g_sig = np.where(g_sig < 0, 0.0, g_sig)
    g_sig2 = g_sig ** 2
    mean, s2 = _max_of_2(g_mu[:, :2], g_sig2[:, :2])
    for i in range(2, g_mu.shape[1]):
        mean, s2 = _max_of_2(np.column_stack([mean, g_mu[:, i]]), np.column_stack([s2, g_sig2[:, i]]))
    std = np.sqrt(s2)
    with np.errstate(all="ignore"):
        d = gbest - mean
        tau = d / std
        return d * norm_cdf(tau) + std * norm_pdf(tau)


def _operator_de(rng, p1, p2, p3, lower, upper, CR=1.0, F=0.5, disM=20.0):
    D = len(p1)
    off = p1.copy()
    site = rng.random(D) < CR
    off[site] += F * (p2[site] - p3[site])
    site, mu = rng.random(D) < 1.0 / D, rng.random(D)
    off = np.minimum(np.maximum(off, lower), upper)
    span = upper - lower
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] += span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lower[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        off[t] += span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (upper[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return off


def _estimate_z(rng, D, lower, upper, models, centers, ref, z):
    delta, nr, max_iter = 0.9, 2, 100
    n = len(ref)
    B = neighbors_of(ref, int(np.ceil(n / 10)))
    pop_x = (upper - lower) * lhs_design(rng, n, D) + lower
    pop_mean = _gp_eval(pop_x, models, centers, False)
    z = np.minimum(pop_mean.min(axis=0), z)
    for _ in range(max_iter - 1):
        for i in range(n):
            P = B[i][rng.permutation(B.shape[1])] if rng.random() < delta else rng.permutation(n)
            off = _operator_de(rng, pop_x[i], pop_x[P[0]], pop_x[P[1]], lower, upper)
            om = _gp_eval(off, models, centers, False)[0]
            z = np.minimum(z, om)
            g_old = np.max((pop_mean[P] - z) * ref[P], axis=1)
            g_new = np.max((om - z) * ref[P], axis=1)
            idx = P[np.where(g_old > g_new)[0][:nr]]
            if len(idx):
                pop_x[idx], pop_mean[idx] = off, om
    return z


def _moead_eti(rng, D, lower, upper, models, centers, ref, gmin, z):
    delta, nr, max_iter = 0.9, 2, 50
    n = len(ref)
    B = neighbors_of(ref, int(np.ceil(n / 10)))
    pop_x = (upper - lower) * lhs_design(rng, n, D) + lower
    pm, ps = _gp_eval(pop_x, models, centers)
    eti = _get_eti(pm, ps, ref, gmin, z)
    for _ in range(max_iter - 1):
        for i in range(n):
            P = B[i][rng.permutation(B.shape[1])] if rng.random() < delta else rng.permutation(n)
            off = _operator_de(rng, pop_x[i], pop_x[P[0]], pop_x[P[1]], lower, upper)
            om, os = _gp_eval(off, models, centers)
            eti_new = _get_eti(np.tile(om, (len(P), 1)), np.tile(os, (len(P), 1)), ref[P], gmin[P], z)
            with np.errstate(invalid="ignore"):
                sel = np.where(eti[P] < eti_new)[0][:nr]
            if len(sel):
                pop_x[P[sel]], pm[P[sel]], ps[P[sel]], eti[P[sel]] = off, om, os, eti_new[sel]
    return eti, pop_x


class MOEADEGO(LoopAlgorithm):
    """Expected-Tchebycheff-improvement (ETI) infill: Kriging models are fitted on fuzzy-c-means neighbourhoods, an
    inner MOEA/D maximises the ETI of every weight vector and a batch of ``batch_size`` promising points is chosen by
    clustering the candidates (best ETI per cluster)."""

    def __init__(self, pop_size: int = 100, batch_size: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.batch_size = int(batch_size)

    def _initialize_infill(self):
        rng, D = self.rng, self.D
        n = 11 * D - 1
        best, best_d = None, -1.0
        for _ in range(1000):                                    # maximin Latin hypercube
            cand = lhs_design(rng, n, D)
            dm = _pdist(cand, cand)
            np.fill_diagonal(dm, np.inf)
            if dm.min() > best_d:
                best, best_d = cand, dm.min()
        return self.evaluate(self.lower + (self.upper - self.lower) * best)

    def _initialize_advance(self, infills=None, **kwargs):
        self.archive = infills
        self.pop = infills[nd_sort(objs(infills), None, 1)[0] == 1]
        self._set_optimum()

    def _ref_vecs(self):
        M = self.M
        if M <= 3:
            return uniform_point(NUM_WEIGHTS[M - 2], M)[0]
        if M <= 6:
            return uniform_point(NUM_WEIGHTS[M - 2], M, "ILD")[0]
        return uniform_point(500, M)[0]

    def step(self):
        rng, D, M = self.rng, self.D, self.M
        batch = min(self.max_FE - self.FE, self.batch_size)
        X, Y = decs(self.archive), objs(self.archive)
        centers, models = _gp_model_fcm(rng, X, Y)
        ref = self._ref_vecs()
        z = _estimate_z(rng, D, self.lower, self.upper, models, centers, ref, Y.min(axis=0))
        G = ref[:, [0]] * (Y[:, 0] - z[0])[None, :]
        for j in range(1, M):
            G = np.maximum(G, ref[:, [j]] * (Y[:, j] - z[j])[None, :])
        gmin = G.min(axis=1)
        eti, cand = _moead_eti(rng, D, self.lower, self.upper, models, centers, ref, gmin, z)
        Q, Qe, temp = [], [], X.copy()
        for i in range(len(cand)):
            if np.min(_pdist(cand[[i]], temp)) > 1e-5 and eti[i] > 0:
                Q.append(cand[i]), Qe.append(eti[i])
                temp = np.vstack([temp, cand[i]])
        Q, Qe = np.array(Q).reshape(-1, D), np.array(Qe)
        b = min(batch, len(Q))
        if b == 0:
            new = cand[rng.permutation(len(cand))[:batch]]
        else:
            lab = kmeans(Q, b, rng)
            pick = []
            for i in range(b):
                idx = np.where(lab == i)[0]
                if len(idx):
                    pick.append(idx[int(np.argmax(Qe[idx]))])
            new = Q[np.array(pick, dtype=int)]
        if len(new) < batch:
            new = np.vstack([new, cand[rng.permutation(len(cand))[: batch - len(new)]]])
        self.archive = Population.merge(self.archive, self.evaluate(new))
        self.pop = self.archive[nd_sort(objs(self.archive), None, 1)[0] == 1]
