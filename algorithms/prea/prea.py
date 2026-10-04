# emopylab 2026
"""PREA (promising-region based EMO algorithm).

Reference:
J. Yuan, H. Liu, F. Gu, Q. Zhang, and Z. He. Investigating the properties of indicators and an
evolutionary many-objective algorithm based on a promising region. IEEE Transactions on Evolutionary
Computation, 2021, 25(1): 75-86.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga_half, objs
from algorithms.icma.icma import _prea_selection
from core.population import Population

ALGORITHM_FLAGS = {'PREA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def indicator_matrix(P):
    """I[i, j]: how much solution j would have to improve (ratio form) to dominate i; negative when it already does."""
    with np.errstate(all="ignore"):
        Ir = P[None, :, :] / P[:, None, :] - 1                     # (i, j, m) = P_j / P_i - 1
        inv = P[:, None, :] / P[None, :, :] - 1                    # (i, j, m) = P_i / P_j - 1
    mx, mn = Ir.max(axis=2), inv.max(axis=2)
    IM = np.where(mx <= 0, -mn, mx)
    np.fill_diagonal(IM, np.inf)
    return IM


def prea_update(pop, N, zmin):
    F = objs(pop)
    P = F - zmin + 1e-6
    IM = indicator_matrix(P)
    ir = IM.min(axis=1)
    lvl1 = np.where(ir >= 0)[0]
    if len(lvl1) <= N:
        idx = np.argsort(-ir, kind="stable")[:N]
        return pop[idx], IM[np.ix_(idx, idx)]
    sel = _prea_selection(P[lvl1], IM[np.ix_(lvl1, lvl1)], N)
    idx = lvl1[sel]
    return pop[idx], IM[np.ix_(idx, idx)]


class PREA(LoopAlgorithm):
    """Ratio-indicator-based many-objective EA: the binary ratio indicator both drives partner selection (each
    solution mates with the solution that comes closest to dominating it) and, together with a distance in the
    reduced objective space, the truncation of the last non-dominated level."""

    def start(self):
        self.zmin = objs(self.pop).min(axis=0)
        self.IM = indicator_matrix(objs(self.pop) - self.zmin + 1e-6)

    def step(self):
        N, rng = self.N, self.rng
        n = len(self.pop)
        neigh = np.argmin(self.IM, axis=1)
        spouse = neigh.copy()
        change = np.where(rng.random(n) > 0.7)[0]
        spouse[change] = np.ceil(rng.random(len(change)) * n).astype(int) - 1
        pool = np.concatenate([np.arange(n), spouse])
        off = self.evaluate(ga_half(self.problem, decs(self.pop[pool]), rng=rng))
        self.zmin = np.minimum(self.zmin, objs(off).min(axis=0))
        self.pop, self.IM = prea_update(Population.merge(self.pop, off), N, self.zmin)
