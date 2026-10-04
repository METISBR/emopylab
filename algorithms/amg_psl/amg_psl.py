# emopylab 2026
"""AMG-PSL (adaptive multi-granular Pareto-optimal subspace learning).

Reference:
C. Sun, Y. Tian, S. Shao, S. Yang, and X. Zhang. An adaptive multi- granular Pareto-optimal subspace
learning algorithm for sparse large- scale multi-objective optimization. Proceedings of the IEEE
Congress on Evolutionary Computation, 2025.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, kmeans, nd_sort, objs, cons, tournament
from algorithms.community_utils.nn import DAE, RBM
from algorithms.community_utils.sparse_mask import spea2_mask_selection
from core.population import Population

ALGORITHM_FLAGS = {'AMGPSL': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


def _cluster_label(fitness_init, labels):
    """Name the cluster with the smaller total score 11 (important variables) and the other 12; also returns the share
    of the smaller-score cluster."""
    n1, n2 = np.sum(labels == 1), np.sum(labels == 2)
    v1, v2 = fitness_init[labels == 1].sum(), fitness_init[labels == 2].sum()
    out = labels.astype(float).copy()
    if v1 < v2:
        rate = n1 / (n1 + n2)
        out[labels == 1], out[labels == 2] = 11, 12
    else:
        rate = n2 / (n1 + n2)
        out[labels == 1], out[labels == 2] = 12, 11
    return rate, out


def _real_crossover(rng, P1, P2, dis_c=20.0):
    n, D = P1.shape
    mu = rng.random((n, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (dis_c + 1)), (2 - 2 * mu) ** (-1 / (dis_c + 1)))
    beta = beta * (-1.0) ** rng.integers(0, 2, (n, D)) * rng.random((n, D))
    return (P1 + P2) / 2 + beta * (P1 - P2) / 2


def _real_mutation(rng, X, lower, upper, dis_m=20.0):
    n, D = X.shape
    lo, up = np.tile(lower, (n, 1)), np.tile(upper, (n, 1))
    site, mu = rng.random((n, D)) < 1.0 / D, rng.random((n, D))
    X = np.minimum(np.maximum(X, lo), up)
    span = up - lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        X[t] += span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (X[t] - lo[t]) / span[t]) ** (dis_m + 1)) ** (1 / (dis_m + 1)) - 1)
        t = site & (mu > 0.5)
        X[t] += span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - X[t]) / span[t]) ** (dis_m + 1)) ** (1 / (dis_m + 1)))
    return np.minimum(np.maximum(X, lo), up)


class AMGPSL(LoopAlgorithm):
    """The variables are layered by a single-probe importance score (layer widths depend on the sparsity level and the
    stage of the run); offspring masks inherit bits from the two parents and are then pushed up/down the layers
    (switching on unused important variables and off used unimportant ones).  A share ``rho`` of the offspring is
    additionally reconstructed through learned mask (RBM) and decision (autoencoder) models."""

    def _initialize_infill(self):
        rng, N, D, enc = self.rng, self.N, self.D, self.encoding
        lo, up = self.lower, self.upper
        real = bool(np.any(enc == 1))
        TDec, TMask, TPop = [], [], []
        if real:
            score = np.zeros((5, D))
            interval = (up - lo) / 5
            for i in range(5):
                for _ in range(2):
                    dec = (lo + interval * i) + rng.random((D, D)) * interval
                    mask = np.eye(D)
                    p = self.evaluate(dec * mask)
                    TDec.append(dec), TMask.append(mask), TPop.append(p)
                    C = cons(p)
                    score[i] += nd_sort(np.hstack([objs(p), C]) if C.size else objs(p), None, np.inf)[0]
            if D > 2000:
                keep = rng.permutation(len(TPop) * D)[:D]
                allp = Population.merge(*TPop)
                TPop, TDec, TMask = [allp[keep]], [np.vstack(TDec)[keep]], [np.vstack(TMask)[keep]]
            init = score.sum(axis=0)
        else:
            dec = np.ones((D, D))
            mask = np.eye(D)
            p = self.evaluate(dec * mask)
            TDec, TMask, TPop = [dec], [mask], [p]
            C = cons(p)
            init = nd_sort(np.hstack([objs(p), C]) if C.size else objs(p), None, np.inf)[0]
        labels = kmeans(init[:, None], 2, rng) + 1
        self.sparse_rate, self.opt_fit = _cluster_label(init, labels)
        pop, self.Dec, self.Mask, self.fit = spea2_mask_selection(Population.merge(*TPop), np.vstack(TDec), np.vstack(TMask), N)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.near_stage = int(np.ceil(self.FE / (self.max_FE / 10)))
        self._update_layer(self.near_stage, None)
        self.rho = 0.5
        self._set_optimum()

    def _update_layer(self, stage, mask):
        D, rng = self.D, self.rng
        group = np.ceil(11 - stage) / 100 * D
        group = np.ceil(self.sparse_rate * 10 * 1 * group)
        if mask is None or mask.sum() == 0:
            order = np.argsort(self.opt_fit + rng.random(D), kind="stable")
        else:
            order = np.argsort(self.opt_fit + (mask == 0).sum(axis=0) / 100000, kind="stable")
        with np.errstate(all="ignore"):
            layer_of_rank = np.ceil(np.arange(1, D + 1) / group)
        layer = np.zeros(D)
        layer[order] = layer_of_rank
        self.layer, self.layer_max = layer, float(layer.max())

    def _train_models(self, Mask, Dec, real):
        rng = self.rng
        M = Mask > 0
        allzero, allone = ~M.any(axis=0), M.all(axis=0)
        other = ~allzero & ~allone
        n_valid = int(other.sum())
        progress = self.FE / self.max_FE
        front, _ = nd_sort(Dec * Mask, np.ones((len(Dec), 1)), np.inf)
        if n_valid:
            vmask, vdec = M[:, other].astype(float), Dec[:, other]
            mean_contrib = np.mean(np.abs(vdec) * vmask, axis=0)
            norm = (mean_contrib - mean_contrib.min()) / (mean_contrib.max() - mean_contrib.min() + np.finfo(float).eps)
            rank_score = np.zeros(len(mean_contrib))
            rank_score[np.argsort(-mean_contrib, kind="stable")] = np.linspace(1, 0, len(mean_contrib))
            best = vmask[front == 1]
            success = best.mean(axis=0) if len(best) else vmask.mean(axis=0)
            var_score = 0.4 * rank_score + 0.4 * success + 0.2 * norm
            thr = var_score.mean() * (1 + 0.2 * (1 - progress))
            typ = var_score > thr
            base_k = min(int(typ.sum()), int(np.round(np.sqrt(n_valid))))
            K = min(max(int(np.round(base_k * (1 - 0.3 * progress))), 1), len(Mask))
        else:
            K = 1
        rbm = dae = None
        if n_valid:
            rbm = RBM(n_valid, K, 10, 1, 0, 0.5, 0.1, rng)
            rbm.train(M[:, other].astype(float))
        if real:
            dae = DAE(Dec.shape[1], K, 10, len(Dec), 0.5, 0.5, 0.1, rng)
            dae.train(Dec)
        return rbm, dae, allzero, allone

    def _operator(self, pdec, pmask, rbm, dae, site, allzero, allone):
        rng, D = self.rng, self.D
        n = len(pdec)
        h = n // 2
        P1m, P2m, P1d, P2d = pmask[:h], pmask[h:], pdec[:h], pdec[h:]
        off = P1m.copy()
        s1, s2 = P1m.sum(axis=1), P2m.sum(axis=1)
        with np.errstate(all="ignore"):
            rate = s1 / (s1 + s2)
        for i in range(h):
            idx = rng.random(D) > rate[i] / 2
            off[i, idx] = P2m[i, idx]
        for i in range(h):
            up, down = 1, self.layer_max
            while up < down:
                if rng.random() < 0.5:
                    tl = np.where(self.layer == up)[0]
                    tgt = tl[off[i, tl] == 0]
                    if len(tgt) and rng.random() < 0.5:
                        off[i, tgt[rng.integers(0, len(tgt), int(np.ceil(len(tgt) / 2)))]] = 1
                    up += 1
                else:
                    tl = np.where(self.layer == down)[0]
                    tgt = tl[off[i, tl] == 1]
                    if len(tgt) and rng.random() < 0.5:
                        off[i, tgt[rng.integers(0, len(tgt), int(np.ceil(len(tgt) / 2)))]] = 0
                    down -= 1
                if rng.random() < 0.5 or up >= down:
                    break
        if np.any(self.encoding != 4):
            rows = np.where(site)[0]
            if len(rows):
                other = ~allzero & ~allone
                if rbm is not None:
                    tmp = off[:, other]
                    for i in rows:
                        tmp[i] = rbm.recover(rbm.reduce(tmp[[i]]))[0]
                    off[:, other] = tmp
                    off[:, allone] = 1
                if dae is not None:
                    off_dec = np.zeros((h, D))
                    for i in rows:
                        off_dec[i] = dae.recover(dae.reduce(P1d[[i]]))[0]
                    rest = np.setdiff1d(np.arange(h), rows)
                    if len(rest):
                        off_dec[rest] = _real_crossover(rng, P1d[rest], P2d[rest])
                else:
                    off_dec = _real_crossover(rng, P1d, P2d)
            else:
                off_dec = _real_crossover(rng, P1d, P2d)
            off_dec = _real_mutation(rng, off_dec, self.lower, self.upper)
            off_dec[:, self.encoding == 4] = 1
        else:
            off_dec = np.ones((h, D))
        return off_dec, off

    def step(self):
        N, rng = self.N, self.rng
        site = self.rho > rng.random(int(np.ceil(N / 2)))
        if site.any():
            rbm, dae, allzero, allone = self._train_models(self.Mask, self.Dec, bool(np.any(self.encoding != 4)))
        else:
            rbm = dae = allzero = allone = None
        mate = tournament(2, 2 * N, self.fit, rng=rng)
        stage = int(np.ceil(self.FE / (self.max_FE / 10)))
        if stage != self.near_stage:
            self.near_stage = stage
            self._update_layer(stage, self.Mask)
        off_dec, off_mask = self._operator(self.Dec[mate], self.Mask[mate], rbm, dae, site, allzero, allone)
        off = self.evaluate(off_dec * off_mask)
        self.pop, self.Dec, self.Mask, self.fit = spea2_mask_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, off_dec]), np.vstack([self.Mask, off_mask]), N)
        # the reference counts successes on an array made of the selected population followed by the offspring
        s1, s2 = 0.0, len(off) / 2
        self.rho = (self.rho + min(max((s1 + 1e-6) / (s1 + s2 + 1e-6), 0.1), 0.9)) / 2
