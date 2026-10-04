# emopylab 2026
"""KLEA (knowledge learning-based evolutionary algorithm).

Reference:
S. Shao, Y. Tian, Y. Zhang, and X. Zhang. Knowledge learning-based dimensionality reduction for
solving large-scale sparse multiobjective optimization problems. IEEE Transactions on Cybernetics,
2025, 55(7): 3471-3484.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, kmeans, objs, tournament
from algorithms.community_utils.nn import LMNet
from algorithms.community_utils.sparse_mask import group_operator_half, lhs_design, spea2_mask_selection
from algorithms.community_utils.spea import cal_fitness
from core.population import Population
from util.hv import hypervolume

ALGORITHM_FLAGS = {'KLEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}

MAX_ACT = 3


def _hv(F, ref):
    F = F[np.all(F <= ref, axis=1)]
    return np.float64(hypervolume(F, ref)) if len(F) else np.float64(0.0)


class KLEA(LoopAlgorithm):
    """A Q-network (trained with Levenberg-Marquardt) picks, from the usage ratio of every variable, how the mask
    variation groups the variables (one variable at a time, blocks of the variable ranking, or two k-means clusters of
    the variable scores); the reward is the relative hypervolume gain of the population."""

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
                fit += cal_fitness(objs(p))
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
        self.Fitness = fit
        pop, self.Dec, self.Mask, self.fit = spea2_mask_selection(Population.merge(*TPop), np.vstack(TDec), np.vstack(TMask), N)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.net = None
        self.action = 1
        self.memory = np.zeros((0, 2 * self.D + 2))
        self._update_memory(self.action, self.pop, self.Mask, self.pop, self.Mask)
        self._set_optimum()

    # -- reinforcement-learning parts ---------------------------------------------------------------------
    def _update_memory(self, action, last_pop, last_mask, pop, mask):
        ref = np.vstack([objs(last_pop), objs(pop)]).max(axis=0)
        h_last, h_new = _hv(objs(last_pop), ref), _hv(objs(pop), ref)
        with np.errstate(all="ignore"):
            reward = (h_new - h_last) / h_last
        reward = 0.0 if not np.isfinite(reward) else float(reward)
        row = np.concatenate([last_mask.sum(axis=0) / len(last_mask), [action, reward], mask.sum(axis=0) / len(mask)])
        self.memory = np.vstack([self.memory, row])
        if len(self.memory) > 1000:
            self.memory = self.memory[[len(self.memory) - 501]]      # (the reference keeps this single row)

    def _threshold(self):
        return int(np.ceil(0.25 * self.max_FE / 100))

    def _using_net(self):
        rng, D = self.rng, self.D
        if rng.random() < 0.3 ** (self.FE / self.max_FE) or len(self.memory) < self._threshold() or self.net is None:
            return int(rng.integers(1, MAX_ACT + 1))
        state = self.memory[-1, D + 2:]
        q = [float(self.net.sim(np.concatenate([state, [j]])[None, :])[0, 0]) for j in range(1, MAX_ACT + 1)]
        return int(np.argmax(q)) + 1

    def _train_net(self):
        D, rng, M = self.D, self.rng, self.memory
        n = len(M)
        X, y = M[:, : D + 1], M[:, D + 1]
        if n == self._threshold():
            self.net = LMNet(rng).configure(X, y)
        if n > self._threshold() and n % 10 == 0 and self.net is not None:
            train = M[rng.permutation(n)[:10]].copy()
            for i in range(len(train)):
                q = [float(self.net.sim(np.concatenate([train[i, D + 2:], [j]])[None, :])[0, 0]) for j in range(1, MAX_ACT + 1)]
                train[i, D + 1] += 1.0 * max(q)
            self.net.train(train[:, : D + 1], train[:, D + 1], epochs=100, goal=1e-3)

    # -- variation ----------------------------------------------------------------------------------------------
    def _operator(self, pdec, pmask):
        rng, D = self.rng, self.D
        n = len(pdec)
        h = n // 2
        P1d, P2d, P1m, P2m = pdec[:h], pdec[h: 2 * h], pmask[:h], pmask[h: 2 * h]
        if np.any(self.encoding != 4):
            off_dec, _, _ = group_operator_half(self, P1d, P2d, 4, rng)
            off_dec[:, self.encoding == 4] = 1
        else:
            off_dec = np.ones(P1d.shape)
        index = np.argsort(self.Fitness, kind="stable") + 1               # rank position -> variable (1-based)
        gsize = int(np.ceil(np.mean(self.Mask) * D))
        recent = self.memory[max(len(self.memory) - 11, 0):, D]
        div = D * len(np.unique(recent)) ** 2
        prob = 100 * np.mean(self.Mask) / div
        if self.action == 1:
            vary = index
        elif self.action == 2:
            vary = np.ceil(index / gsize).astype(int)
        else:
            vary = kmeans(self.Fitness[:, None], 2, rng) + 1
        max_g = int(vary.max())
        off = P1m.copy()
        for i in range(h):
            sel = int(rng.integers(1, max_g + 1))
            diff = (P1m[i] > 0) ^ (P2m[i] > 0)
            idx = (vary == sel) & diff & (rng.random(D) < 1)
            off[i, idx] = 0 if rng.random() < 0.5 else 1
        for i in range(h):
            direction = 0 if rng.random() < 0.5 else 1
            off[i, rng.random(D) < prob] = direction
        return off_dec, off

    def step(self):
        N, rng = self.N, self.rng
        mate = tournament(2, 2 * N, self.fit, rng=rng)
        last_pop, last_mask = self.pop, self.Mask
        off_dec, off_mask = self._operator(self.Dec[mate], self.Mask[mate])
        off = self.evaluate(off_dec * off_mask)
        self.pop, self.Dec, self.Mask, self.fit = spea2_mask_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, off_dec]), np.vstack([self.Mask, off_mask]), N)
        self._update_memory(self.action, last_pop, last_mask, self.pop, self.Mask)
        self.action = self._using_net()
        self._train_net()
