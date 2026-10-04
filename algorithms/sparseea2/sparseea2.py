# emopylab 2026
"""SparseEA2 (improved SparseEA).

Reference:
Y. Zhang, Y. Tian, and X. Zhang. Improved SparseEA for sparse large-scale multi-objective
optimization problems. Complex & Intelligent Systems, 2023, 9: 1127-1142.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, tournament
from algorithms.sparseea.sparseea import _env_selection, probe_and_init
from core.population import Population

ALGORITHM_FLAGS = {'SparseEA2': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


def _glp_ga_half(algo, P1, P2, groups):
    """SBX between the two parent halves followed by polynomial mutation of one randomly chosen variable group
    per solution (groups are formed by ranking the offspring's variable values)."""
    rng = algo.rng
    N, D = P1.shape
    lo, up = algo.lower, algo.upper
    mu = rng.random((N, D))
    beta = np.zeros((N, D))
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / 21)
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / 21)
    beta = beta * (-1.0) ** rng.integers(0, 2, (N, D))
    beta[rng.random((N, D)) < 0.5] = 1
    off = (P1 + P2) / 2 + beta * (P1 - P2) / 2
    per = D // groups
    gidx = np.ones((N, D), dtype=int)
    for s in range(N):
        order = np.argsort(off[s], kind="stable")
        for g in range(1, groups):
            gidx[s, order[(g - 1) * per:g * per]] = g
        gidx[s, order[(groups - 1) * per:]] = groups
    chosen = rng.integers(1, groups + 1, size=N)
    site = gidx == chosen[:, None]
    mu = np.repeat(rng.random((N, 1)), D, axis=1)
    Lo, Up = np.tile(lo, (N, 1)), np.tile(up, (N, 1))
    off = np.minimum(np.maximum(off, Lo), Up)
    span = Up - Lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - Lo[t]) / span[t]) ** 21) ** (1 / 21) - 1)
        t = site & (mu > 0.5)
        off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (Up[t] - off[t]) / span[t]) ** 21) ** (1 / 21))
    return off, gidx, chosen


def _ts(rng, f):
    return None if len(f) == 0 else int(tournament(2, 1, f, rng=rng)[0])


class SparseEA2(LoopAlgorithm):
    """SparseEA with group-wise variation: the decision vector is mutated only inside one randomly chosen group of
    similarly valued variables, and the mask flips of the second phase are restricted to the same group."""

    def _initialize_infill(self):
        P, TPop, Dec, Mask, TDec, TMask, self.fitness = probe_and_init(self)
        pop, self.Dec, self.Mask, self.front, self.crowd = _env_selection(
            Population.merge(P, *TPop), np.vstack([Dec] + TDec), np.vstack([Mask] + TMask), self.N)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _operator(self, ParentDec, ParentMask):
        rng, Fit = self.rng, self.fitness
        n = len(ParentDec)
        h = n // 2
        P1d, P2d = ParentDec[:h], ParentDec[h:2 * h]
        P1m, P2m = ParentMask[:h], ParentMask[h:2 * h]
        real = bool(np.any(self.encoding != 4))
        if real:
            OffDec, gidx, chosen = _glp_ga_half(self, P1d, P2d, 4)
            OffDec[:, self.encoding == 4] = 1
        else:
            OffDec = np.ones(P1d.shape)
        Off = P1m.copy()
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where((P1m[i] != 0) & (P2m[i] == 0))[0]
                k = _ts(rng, -Fit[idx])
                if k is not None:
                    Off[i, idx[k]] = 0
            else:
                idx = np.where((P1m[i] == 0) & (P2m[i] != 0))[0]
                k = _ts(rng, Fit[idx])
                if k is not None:
                    Off[i, idx[k]] = P2m[i, idx[k]]
        if real:
            chosen_idx = gidx == chosen[:, None]
            for i in range(h):
                if rng.random() < 0.5:
                    idx = np.where((Off[i] != 0) & chosen_idx[i])[0]
                    k = _ts(rng, -Fit[idx])
                    if k is not None:
                        Off[i, idx[k]] = 0
                else:
                    idx = np.where((Off[i] == 0) & chosen_idx[i])[0]
                    k = _ts(rng, Fit[idx])
                    if k is not None:
                        Off[i, idx[k]] = 1
        return OffDec, Off

    def step(self):
        pool = tournament(2, 2 * self.N, self.front, -self.crowd, rng=self.rng)
        OffDec, OffMask = self._operator(self.Dec[pool], self.Mask[pool])
        off = self.evaluate(OffDec * OffMask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd = _env_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), self.N)
