# emopylab 2026
"""NNIA (nondominated neighbor immune algorithm).

Reference:
M. Gong, L. Jiao, H. Du, and L. Bo. Multiobjective immune algorithm with nondominated neighbor-based
selection. Evolutionary Computation, 2008, 16(2): 225-255.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, ga_half, decs, nd_sort, objs

ALGORITHM_FLAGS = {'NNIA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class NNIA(LoopAlgorithm):
    """Dominant population D (nondominated, crowding-truncated) is cloned in proportion to
    crowding distance and varied by SBX/PM against randomly chosen active antibodies."""

    def __init__(self, pop_size: int = 100, nA: int = 20, nC: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.nA = int(nA)
        self.nC = int(nC)

    def _update_dominant(self, P):
        front_no, _ = nd_sort(objs(P), None, 1)
        P = P[np.where(front_no == 1)[0]]
        rank = np.argsort(-crowding(objs(P)), kind="stable")
        return P[rank[: min(self.N, len(rank))]]

    def _cloning(self, A):
        cd = crowding(objs(A))
        if np.all(np.isinf(cd)):
            cd = np.ones_like(cd)
        else:
            finite = np.isfinite(cd)
            cd = cd.copy()
            cd[~finite] = 2.0 * np.max(cd[finite])
        q = np.ceil(self.nC * cd / np.sum(cd)).astype(int)
        return A[np.repeat(np.arange(len(A)), q)]

    def start(self):
        self.pop = self._update_dominant(self.pop)

    def step(self):
        D = self.pop
        A = D[: min(self.nA, len(D))]
        C = self._cloning(A)
        partners = A[self.rng.integers(0, len(A), size=len(C))]
        off = ga_half(self.problem, np.vstack([decs(C), decs(partners)]), rng=self.rng)
        C1 = self.evaluate(off)
        from core.population import Population
        self.pop = self._update_dominant(Population.merge(D, C1))
