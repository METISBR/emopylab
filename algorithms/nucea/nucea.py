# emopylab 2026
"""NUCEA (non-uniform clustering based evolutionary algorithm).

Reference:
S. Shao, Y. Tian, and X. Zhang. A non-uniform clustering based evolutionary algorithm for solving
large-scale sparse multi-objective optimization problems. Proceedings of the 18th International
Conference on Bio-inspired Computing: Theories and Applications, 2023.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, tournament
from algorithms.community_utils.sparse_mask import group_operator_half, probe_variables, spea2_mask_selection, ts
from core.population import Population

ALGORITHM_FLAGS = {'NUCEA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class NUCEA(LoopAlgorithm):
    """The variables are ranked by their single-probe score and cut into groups of ``ceil(mean sparsity * D)``
    consecutive variables; offspring masks only change (remove/add) the variables of one randomly chosen group where the
    parents disagree, and real parts are varied group by group."""

    def _initialize_infill(self):
        rng, N, D, enc = self.rng, self.N, self.D, self.encoding
        Dec0, Mask0, Pop0, score = probe_variables(self)
        self.Fitness = score
        Mask = np.zeros((N, D))
        for i in range(N):
            Mask[i, tournament(2, int(np.ceil(rng.random() * D)), score, rng=rng)] = 1
        Dec = self.lower + rng.random((N, D)) * (self.upper - self.lower)
        Dec[:, enc == 4] = 1
        pop = self.evaluate(Dec * Mask)
        pop, self.Dec, self.Mask, self.fit = spea2_mask_selection(Population.merge(pop, *Pop0), np.vstack([Dec] + Dec0), np.vstack([Mask] + Mask0), N)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _operator(self, pdec, pmask):
        rng, D = self.rng, self.D
        n = len(pdec)
        h = n // 2
        P1d, P2d = pdec[:h], pdec[h: 2 * h]
        P1m, P2m = pmask[:h], pmask[h: 2 * h]
        order = np.argsort(self.Fitness, kind="stable")
        gsize = int(np.ceil(np.mean(self.Mask) * D))
        vary = np.ones(D, dtype=int)
        start, g = 0, 1
        while True:
            end = min(start + g * gsize, D)
            vary[order[start:end]] = g
            g += 1
            start = end
            if start >= D:
                break
        max_g = int(vary.max())
        real = bool(np.any(self.encoding != 4))
        if real:
            off_dec, groups, chosen = group_operator_half(self, P1d, P2d, 4, rng)
            off_dec[:, self.encoding == 4] = 1
        else:
            off_dec = np.ones(P1d.shape)
        off = P1m.copy()
        sel = 1
        for i in range(h):
            sel = int(rng.integers(1, max_g + 1))
            diff = (P1m[i] > 0) ^ (P2m[i] > 0)
            idx = (vary == sel) & diff
            off[i, idx] = 0 if rng.random() < 0.5 else 1
        if real and sel < max_g:
            inside = groups == chosen
            for i in range(h):
                if rng.random() < 0.5:
                    idx = np.where((off[i] > 0) & inside[i])[0]
                    t = ts(-self.Fitness[idx], rng)
                    if t is not None:
                        off[i, idx[t]] = 0
                else:
                    idx = np.where((off[i] == 0) & inside[i])[0]
                    t = ts(self.Fitness[idx], rng)
                    if t is not None:
                        off[i, idx[t]] = 1
        return off_dec, off

    def step(self):
        N, rng = self.N, self.rng
        mate = tournament(2, 2 * N, self.fit, rng=rng)
        off_dec, off_mask = self._operator(self.Dec[mate], self.Mask[mate])
        off = self.evaluate(off_dec * off_mask)
        self.pop, self.Dec, self.Mask, self.fit = spea2_mask_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, off_dec]), np.vstack([self.Mask, off_mask]), N)
