# emopylab 2026
"""DM-MOEA (dual model based multi-objective evolutionary algorithm).

Reference:
P. Zhang, R. Zhang, Y. Tian, K. C. Tan, and X. Zhang. A dual model-based evolutionary framework for
dynamic large-scale sparse multiobjective optimization. Swarm and Evolutionary Computation, 2025,
97: 102011.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, objs, tournament
from algorithms.community_utils.sparse_mask import group_operator_half, nsga2_mask_selection, probe_variables, ts
from algorithms.kl_nsga_ii.kl_nsga_ii import changed
from core.population import Population
from util.svm import SVR

ALGORITHM_FLAGS = {'DMMOEA': {'binary', 'constrained', 'dynamic', 'integer', 'large', 'multi', 'real', 'sparse'}}

P_HISTORY = 4


def _sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def _mlp_predict(rng, data, score):
    """Tiny one-hidden-layer network (10 sigmoid units, 500 epochs, rate 0.1) fitted to map the earlier states onto the
    latest one, evaluated on the latest state, for the variables with a non-zero score."""
    po = np.where(score != 0)[0]
    pre = np.zeros(data.shape[1])
    d = data[:, po]
    train, target = d[:-1], d[-1:]
    n_in, n_hid, n_out = train.shape[1], 10, target.shape[1]
    W1, W2 = rng.standard_normal((n_in, n_hid)), rng.standard_normal((n_hid, n_out))
    b1, b2 = rng.standard_normal((1, n_hid)), rng.standard_normal((1, n_out))
    lr = 0.1
    for _ in range(500):
        h = _sigmoid(train @ W1 + b1)
        out = _sigmoid(h @ W2 + b2)
        d_out = out - target
        d_hid = (d_out @ W2.T) * h * (1 - h)
        W2 -= lr * (h.T @ d_out)
        b2 -= lr * d_out.sum(axis=0, keepdims=True)
        W1 -= lr * (train.T @ d_hid)
        b1 -= lr * d_hid.sum(axis=0, keepdims=True)
    pre[po] = _sigmoid(_sigmoid(target @ W1 + b1) @ W2 + b2)[0]
    return pre


class DMMOEA(LoopAlgorithm):
    """NUCEA-style mask search for dynamic problems: when a change is detected the non-dominated solutions are
    re-evaluated, the masks are predicted from their history with support vector regression (after enough changes) and the
    decision values with a small neural network, and the populations of finished environments are kept as the result."""

    def _initialize_infill(self):
        rng, N, D, enc = self.rng, self.N, self.D, self.encoding
        Dec0, Mask0, Pop0, score = probe_variables(self)
        self.Fitness = score
        Dec = self.lower + rng.random((N, D)) * (self.upper - self.lower)
        Dec[:, enc == 4] = 1
        Mask = np.zeros((N, D))
        for i in range(N):
            Mask[i, tournament(2, int(np.ceil(rng.random() * D)), score, rng=rng)] = 1
        pop = self.evaluate(Dec * Mask)
        self.dec_source, self.mask_source = [Dec], [Mask]
        pop_all = Population.merge(pop, *Pop0)
        pop, self.Dec, self.Mask, self.front, self.crowd = nsga2_mask_selection(pop_all, np.vstack([Dec] + Dec0), np.vstack([Mask] + Mask0), N)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.change_count, self.all_pop = 0, None
        self._set_optimum()

    # -- prediction -----------------------------------------------------------------------------------------
    def _svr_masks(self, cc):
        P = P_HISTORY
        mask = self.mask_source[cc].copy()
        rng = self.rng
        for i in range(len(mask)):
            data = np.vstack([self.mask_source[cc - P + j - 1][i] for j in range(1, P + 2)])
            score = data.mean(axis=0)
            idx1, idx2 = np.where(score == 1)[0], np.where(score == 0)[0]
            idx3 = np.setdiff1d(np.arange(data.shape[1]), np.concatenate([idx1, idx2]))
            d = data[:, idx3]
            Xtr, Ytr, Xte = d[:-1], d[1:], d[-1:]
            pred = np.zeros(len(idx3))
            for k in range(len(idx3)):
                m = SVR(kernel_scale=1.0, standardize=False).fit(Xtr, Ytr[:, k])
                pred[k] = float(m.predict(Xte)[0] > rng.random())
            row = np.zeros(data.shape[1])
            row[idx1], row[idx2], row[idx3] = 1, 0, pred
            mask[i] = row
        return mask

    def _mlp_decisions(self, cc, score):
        P = P_HISTORY
        dec = self.dec_source[cc].copy()
        for i in range(len(dec)):
            if cc < P:
                data = np.vstack([self.dec_source[j][i] for j in range(0, cc + 1)])
            else:
                data = np.vstack([self.dec_source[cc - P + j - 1][i] for j in range(1, P + 2)])
            dec[i] = _mlp_predict(self.rng, data, score)
        return dec

    def _prediction(self, cc):
        mask = self.mask_source[cc].copy() if cc < P_HISTORY else self._svr_masks(cc)
        dec = self._mlp_decisions(cc, mask.sum(axis=0))
        return self.evaluate(dec * mask), dec, mask

    # -- variation --------------------------------------------------------------------------------------------
    def _operator(self, pdec, pmask):
        rng, D, Fit = self.rng, self.D, self.Fitness
        h = len(pdec) // 2
        P1d, P2d, P1m, P2m = pdec[:h], pdec[h: 2 * h], pmask[:h], pmask[h: 2 * h]
        real = bool(np.any(self.encoding != 4))
        if real:
            off_dec, groups, chosen = group_operator_half(self, P1d, P2d, 4, rng)
            off_dec[:, self.encoding == 4] = 1
        else:
            off_dec = np.ones(P1d.shape)
        off = P1m.copy()
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where((P1m[i] > 0) & (P2m[i] == 0))[0]
                t = ts(-Fit[idx], rng)
                if t is not None:
                    off[i, idx[t]] = 0
            else:
                idx = np.where((P1m[i] == 0) & (P2m[i] > 0))[0]
                t = ts(Fit[idx], rng)
                if t is not None:
                    off[i, idx[t]] = P2m[i, idx[t]]
        if real:
            inside = groups == chosen
            for i in range(h):
                if rng.random() < 0.5:
                    idx = np.where((off[i] > 0) & inside[i])[0]
                    t = ts(-Fit[idx], rng)
                    if t is not None:
                        off[i, idx[t]] = 0
                else:
                    idx = np.where((off[i] == 0) & inside[i])[0]
                    t = ts(Fit[idx], rng)
                    if t is not None:
                        off[i, idx[t]] = 1
        return off_dec, off

    def step(self):
        N, rng = self.N, self.rng
        if changed(self, self.pop):
            keep = self.front == 1
            pdec, pmask = self.Dec[keep], self.Mask[keep]
            renewed = self.evaluate(pdec * pmask)
            self.change_count += 1
            cc = self.change_count
            self.dec_source.append(self.Dec)
            self.mask_source.append(self.Mask)
            self.all_pop = self.pop if self.all_pop is None else Population.merge(self.all_pop, self.pop)
            pred, dec, mask = self._prediction(cc)
            self.pop, self.Dec, self.Mask, self.front, self.crowd = nsga2_mask_selection(
                Population.merge(pred, renewed), np.vstack([dec, pdec]), np.vstack([mask, pmask]), N)
        mate = tournament(2, 2 * N, self.front, -self.crowd, rng=rng)
        off_dec, off_mask = self._operator(self.Dec[mate], self.Mask[mate])
        off = self.evaluate(off_dec * off_mask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd = nsga2_mask_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, off_dec]), np.vstack([self.Mask, off_mask]), N)
        if self.FE >= self.max_FE and self.all_pop is not None:
            self.pop = Population.merge(self.all_pop, self.pop)
