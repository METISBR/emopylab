# emopylab 2026
"""MOBCA (multi-objective besiege and conquer algorithm).

Reference:
J. Jiang, J. Wu, J. Luo, X. Yang, and Z. Huang. MOBCA: multi-objective besiege and conquer
algorithm. Biomimetics, 2024, 9: 316.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, objs
from algorithms.mopso.mopso import _rep_selection, _update_archive, _delete
from algorithms.community_utils.base import first_front
from core.population import Population

ALGORITHM_FLAGS = {'MOBCA': {'integer', 'multi', 'real'}}


def _bca_update_pop(archive, pop, n_armies, div, rng):
    new = Population.merge(archive, pop)
    nd = new[first_front(objs(new))]
    if len(nd) > n_armies:
        nd = nd[~_delete(objs(nd), len(nd) - n_armies, div, rng)]
    if len(nd) >= n_armies:
        return nd[:n_armies]
    flag = rng.random(n_armies) < 0.1
    idx = np.where(flag)[0]
    m = min(len(nd), len(idx))
    out = pop.copy(deep=False)
    out[idx[rng.permutation(m)]] = nd[:m]
    return out


class MOBCA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, BCB: float = 0.2, nSoldiers: int = 3, div: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.BCB, self.nS, self.div = float(BCB), int(nSoldiers), int(div)

    def initial_size(self):
        self.nA = int(self.pop_size // self.nS)
        return self.nA

    def start(self):
        self.armies = self.pop
        self.archive = self.pop

    def _operator(self, parent, archive_sel):
        rng, nA, nS, D = self.rng, self.nA, self.nS, self.D
        P, A = np.asarray(parent.get("X"), float), np.asarray(archive_sel.get("X"), float)
        r = rng.integers(0, nA - 1, size=(nA, nS))
        r = np.where(r >= np.arange(nA)[:, None], r + 1, r)                         # a different army
        Pi, Pr = P[:, None, :], P[r]                                                # (nA,1,D), (nA,nS,D)
        gap = np.abs(Pr - Pi)
        use_a = rng.random((nA, nS, D)) < self.BCB
        alpha, beta = rng.random((nA, nS, D)) * 2 * np.pi, rng.random((nA, nS, D)) * 2 * np.pi
        soldiers = np.where(use_a, A[:, None, :] + gap * np.sin(alpha), Pr + gap * np.cos(beta))
        return self.evaluate(soldiers.reshape(nA * nS, D))

    def step(self):
        rep = _rep_selection(objs(self.archive), self.nA, self.div, self.rng)
        off = self._operator(self.armies, self.archive[rep])
        self.archive = _update_archive(Population.merge(self.archive, off), self.N, self.div, self.rng)
        self.armies = _bca_update_pop(self.archive, self.armies, self.nA, self.div, self.rng)
        self.pop = self.archive
