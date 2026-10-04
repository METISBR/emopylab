# emopylab 2026
"""DirHV-EI (expected direction-based hypervolume improvement).

Reference:
L. Zhao and Q. Zhang. Hypervolume-guided decomposition for parallel expensive multiobjective
optimization. IEEE Transactions on Evolutionary Computation, 2024, 28(2): 432-444.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, neighbors_of, objs
from algorithms.community_utils.dace import DaceModel, norm_cdf, norm_pdf
from algorithms.community_utils.sparse_mask import lhs_design
from algorithms.moea_d_ego.moea_d_ego import NUM_WEIGHTS, _operator_de, _pdist
from algorithms.community_utils.base import uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'DirHVEI': {'expensive', 'integer', 'many', 'multi', 'real'}}


def _get_xis(nds, ref, z):
    temp = 1.1 * ref - z
    dirs = temp / np.linalg.norm(temp, axis=1, keepdims=True)
    div = 1.0 / dirs
    T = nds - z
    G = div[:, [0]] * T[:, 0][None, :]
    for j in range(1, ref.shape[1]):
        G = np.maximum(G, div[:, [j]] * T[:, j][None, :])
    return z + G.min(axis=1)[:, None] * dirs, dirs


def _dirhvei(u, sigma, xis):
    d = xis - u
    with np.errstate(all="ignore"):
        tau = d / sigma
        return np.prod(d * norm_cdf(tau) + sigma * norm_pdf(tau), axis=1)


def _gp_eval(X, models):
    X = np.atleast_2d(X)
    preds = [m.predict(X, mse=True) for m in models]
    u = np.column_stack([p[0] for p in preds])
    mse = np.column_stack([p[1] for p in preds])
    mse[mse < 0] = 0
    return u, np.sqrt(mse)


def _subset_selection(H, batch):
    L, N = H.shape
    qb, beta, temp = [], np.zeros(N), H.copy()
    for _ in range(batch):
        idx = int(np.argmax(temp.sum(axis=1)))
        qb.append(idx)
        beta = beta + temp[idx]
        temp = np.maximum(H - beta, 0)
    return np.array(qb)


class DirHVEI(LoopAlgorithm):
    """Kriging models on the normalised objectives; the infill criterion is the expected improvement of the hypervolume
    dominated along a set of directions (one value per direction), maximised by an inner MOEA/D, and a batch is chosen by
    greedy submodular subset selection over the directions."""

    def __init__(self, pop_size: int = 100, batch_size: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.batch_size = int(batch_size)

    def _initialize_infill(self):
        rng, D = self.rng, self.D
        n = 11 * D - 1
        best, best_d = None, -1.0
        for _ in range(1000):
            cand = lhs_design(rng, n, D)
            dm = _pdist(cand, cand)
            np.fill_diagonal(dm, np.inf)
            if dm.min() > best_d:
                best, best_d = cand, dm.min()
        return self.evaluate(self.lower + (self.upper - self.lower) * best)

    def _initialize_advance(self, infills=None, **kwargs):
        self.archive = infills
        self.pop = infills[nd_sort(objs(infills), None, 1)[0] == 1]
        n = 11 * self.D - 1
        self.theta = [n ** (-1.0 / n) * np.ones(self.D) for _ in range(self.M)]
        self._set_optimum()

    def _ref_vecs(self):
        M = self.M
        if M <= 3:
            return uniform_point(NUM_WEIGHTS[M - 2], M)[0]
        if M <= 6:
            return uniform_point(NUM_WEIGHTS[M - 2], M, "ILD")[0]
        return uniform_point(500, M)[0]

    def _moead_gr(self, models, dirs, xis):
        rng, D = self.rng, self.D
        lower, upper = self.lower, self.upper
        n = len(dirs)
        B = neighbors_of(dirs, int(np.ceil(n / 10)))
        pop_x = (upper - lower) * lhs_design(rng, n, D) + lower
        pm, ps = _gp_eval(pop_x, models)
        val = _dirhvei(pm, ps, xis)
        for _ in range(49):
            for i in range(n):
                P = B[i][rng.permutation(B.shape[1])] if rng.random() < 0.8 else rng.permutation(n)
                off = _operator_de(rng, pop_x[i], pop_x[P[0]], pop_x[P[1]], lower, upper)
                om, os = _gp_eval(off, models)
                v = _dirhvei(np.tile(om, (n, 1)), np.tile(os, (n, 1)), xis)
                best = int(np.argmax(np.where(np.isnan(v), -np.inf, v)))
                P = B[best]
                with np.errstate(invalid="ignore"):
                    sel = P[val[P] < v[P]]
                if len(sel):
                    pop_x[sel], pm[sel], ps[sel], val[sel] = off, om, os, v[sel]
        return pop_x, pm, ps

    def step(self):
        rng, D, M = self.rng, self.D, self.M
        batch = min(self.max_FE - self.FE, self.batch_size)
        X, F = decs(self.archive), objs(self.archive)
        ymin, ymax = F.min(axis=0), F.max(axis=0)
        with np.errstate(all="ignore"):
            Y = (F - ymin) / (ymax - ymin)
        nds = Y[nd_sort(F, None, 1)[0] == 1]
        models = []
        for i in range(M):
            dm = DaceModel(X, Y[:, i], "regpoly0", self.theta[i], 1e-6 * np.ones(D), 20 * np.ones(D))
            models.append(dm)
            self.theta[i] = dm.theta
        ref = self._ref_vecs()
        z = -0.01 * np.ones(M)
        xis, dirs = _get_xis(nds, ref, z)
        cx, cm, cs = self._moead_gr(models, dirs, xis)
        cx, ia = np.unique(cx, axis=0, return_index=True)
        cm, cs = cm[ia], cs[ia]
        H = np.array([_dirhvei(np.tile(cm[j], (len(dirs), 1)), np.tile(cs[j], (len(dirs), 1)), xis) for j in range(len(cx))])
        H = np.nan_to_num(H)
        new = cx[_subset_selection(H, batch)]
        self.archive = Population.merge(self.archive, self.evaluate(new))
        self.pop = self.archive[nd_sort(objs(self.archive), None, 1)[0] == 1]
