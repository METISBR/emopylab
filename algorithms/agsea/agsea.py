# emopylab 2026
"""AGSEA (automated guiding vector selection-based evolutionary algorithm).

Reference:
S. Shao, Y. Tian, and X. Zhang. Deep reinforcement learning assisted automated guiding vector
selection for large-scale sparse multi-objective optimization. Swarm and Evolutionary Computation,
2024, 88: 101606.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, kmeans, objs, tournament
from algorithms.community_utils.nn import LMNet
from algorithms.community_utils.sparse_mask import group_operator_half, lhs_design, spea2_mask_selection
from core.population import Population
from util.hv import hypervolume

ALGORITHM_FLAGS = {'AGSEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}

NUM_FEATURE = 14
MAX_ACT = 3


def _hv(F, ref):
    F = F[np.all(F <= ref, axis=1)]
    return np.float64(hypervolume(F, ref)) if len(F) else np.float64(0.0)


def _histcounts_01(x):
    """Counts of ``x`` in the bins [0,.1), ..., [.9,1] (the last bin is closed)."""
    return np.histogram(x, bins=np.linspace(0, 1, 11))[0].astype(float)


class AGSEA(LoopAlgorithm):
    """A Q-network trained with Levenberg-Marquardt reads a summary of the population sparsity (mean, spread and the
    histogram of the mask densities, plus the progress of the run) and picks the score that ranks the variables for the
    two-cluster grouping of the mask variation: the single-variable probe score, the number of zeros in the masks, or
    random scores."""

    def _initialize_infill(self):
        rng, N, D, enc = self.rng, self.N, self.D, self.encoding
        lo, up = self.lower, self.upper
        real = bool(np.any(enc == 1))
        TDec, TMask, TPop = [], [], []
        fit = np.zeros(D)
        if real:
            DecMat = (up - lo) * lhs_design(rng, 5, D) - lo
            for i in range(5):
                dec = np.tile(DecMat[i], (D, 1))
                mask = np.eye(D)
                p = self.evaluate(dec * mask)
                TDec.append(dec), TMask.append(mask), TPop.append(p)
                fit += objs(p).sum(axis=1)
            if D > 0:
                keep = rng.permutation(D * 5)[:D]
                allp = Population.merge(*TPop)
                TPop, TDec, TMask = [allp[keep]], [np.vstack(TDec)[keep]], [np.vstack(TMask)[keep]]
        else:
            dec = np.ones((D, D))
            mask = np.eye(D)
            p = self.evaluate(dec * mask)
            TDec.append(dec), TMask.append(mask), TPop.append(p)
            fit += objs(p).sum(axis=1)
        dec = lo + rng.random((N, D)) * (up - lo)
        dec[:, enc == 4] = 1
        mask = np.zeros((N, D))
        for i in range(N):
            mask[i, tournament(2, int(np.ceil(rng.random() * D)), fit, rng=rng)] = 1
        p = self.evaluate(dec * mask)
        TDec.append(dec), TMask.append(mask), TPop.append(p)
        self.Fitness1 = fit
        pop, self.Dec, self.Mask, self.fit = spea2_mask_selection(Population.merge(*TPop), np.vstack(TDec), np.vstack(TMask), N)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.net = None
        self.action = 1
        self.memory = np.zeros((0, 2 * (NUM_FEATURE - 1) + 2))
        self._update_memory(self.action, self.pop, self.Mask, self.pop, self.Mask)
        self._set_optimum()

    def _update_memory(self, action, last_pop, last_mask, pop, mask):
        ref = np.vstack([objs(last_pop), objs(pop)]).max(axis=0)
        h_last, h_new = _hv(objs(last_pop), ref), _hv(objs(pop), ref)
        with np.errstate(all="ignore"):
            reward = (h_new - h_last) / h_last
        reward = 0.0 if not np.isfinite(reward) else float(reward)
        last_state = np.concatenate([[np.mean(last_mask), np.std(last_mask.sum(axis=1), ddof=1) if len(last_mask) > 1 else 0.0],
                                     _histcounts_01(last_mask.sum(axis=1) / last_mask.shape[1]) / last_mask.shape[1], [self.FE / self.max_FE]])
        cur_state = np.concatenate([[np.mean(mask), np.std(last_mask.sum(axis=1), ddof=1) if len(last_mask) > 1 else 0.0],
                                    _histcounts_01(mask.sum(axis=1) / mask.shape[1]) / mask.shape[1], [self.FE / self.max_FE]])
        self.memory = np.vstack([self.memory, np.concatenate([last_state, [action, reward], cur_state])])

    def _threshold(self):
        return int(np.ceil(0.25 * self.max_FE / 100))

    def _using_net(self):
        rng = self.rng
        n = len(self.memory)
        if n == 1:
            pass
        elif rng.random() < 0.3 ** (self.FE / self.max_FE) or n < self._threshold() or self.net is None:
            self.action = int(rng.integers(1, MAX_ACT + 1))
        else:
            state = self.memory[-1, NUM_FEATURE + 1:]
            q = [float(self.net.sim(np.concatenate([state, [j]])[None, :])[0, 0]) for j in range(1, MAX_ACT + 1)]
            self.action = int(np.argmax(q)) + 1
        if self.action == 1:
            return self.Fitness1
        if self.action == 2:
            return (self.Mask == 0).sum(axis=0).astype(float) if len(self.Mask) > 1 else self.Mask[0].astype(float)
        return rng.random(self.D)

    def _train_net(self):
        rng, M, nf = self.rng, self.memory, NUM_FEATURE
        n = len(M)
        if n == self._threshold():
            self.net = LMNet(rng).configure(M[:, :nf], M[:, nf])
        if n > self._threshold() and n % 10 == 0 and self.net is not None:
            train = M[-10:].copy()
            for i in range(len(train)):
                q = [float(self.net.sim(np.concatenate([train[i, nf + 1:], [j]])[None, :])[0, 0]) for j in range(1, MAX_ACT + 1)]
                train[i, nf] += 0.1 * max(q)
            self.net.train(train[:, :nf], train[:, nf], epochs=100, goal=1e-3)

    def _operator(self, pdec, pmask, fitness):
        rng, D = self.rng, self.D
        n = len(pdec)
        h = n // 2
        P1d, P2d, P1m, P2m = pdec[:h], pdec[h: 2 * h], pmask[:h], pmask[h: 2 * h]
        vary = kmeans(fitness[:, None], 2, rng) + 1
        max_g = int(vary.max())
        if np.any(self.encoding != 4):
            off_dec, _, _ = group_operator_half(self, P1d, P2d, 4, rng)
            off_dec[:, self.encoding == 4] = 1
        else:
            off_dec = np.ones(P1d.shape)
        off = P1m.copy()
        for i in range(h):
            sel = int(rng.integers(1, max_g + 1))
            diff = (P1m[i] > 0) ^ (P2m[i] > 0)
            idx = (vary == sel) & diff & (rng.random(D) < 1)
            off[i, idx] = 0 if rng.random() < 0.5 else 1
        recent = self.memory[max(len(self.memory) - 11, 0):, NUM_FEATURE - 1]
        rate = 100 * np.mean(self.Mask) / (D * len(np.unique(recent)) ** 2)
        for i in range(h):
            direction = 0 if rng.random() < 0.5 else 1
            off[i, rng.random(D) < rate] = direction
        return off_dec, off

    def step(self):
        N, rng = self.N, self.rng
        mate = tournament(2, 2 * N, self.fit, rng=rng)
        last_pop, last_mask = self.pop, self.Mask
        fitness = self._using_net()
        off_dec, off_mask = self._operator(self.Dec[mate], self.Mask[mate], fitness)
        off = self.evaluate(off_dec * off_mask)
        self.pop, self.Dec, self.Mask, self.fit = spea2_mask_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, off_dec]), np.vstack([self.Mask, off_mask]), N)
        self._update_memory(self.action, last_pop, last_mask, self.pop, self.Mask)
        self._train_net()
