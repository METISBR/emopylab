# emopylab 2026
"""I-DBEA (improved decomposition-based evolutionary algorithm).

Reference:
M. Asafuddoula, T. Ray, and R. Sarker. A decomposition-based evolutionary algorithm for many
objective optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(3): 445-460.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, cv, decs, ga_half, objs, uniform_point

ALGORITHM_FLAGS = {'IDBEA': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _intercepts(F):
    N, M = F.shape
    choosed = list(np.argmin(F, axis=0))
    l2 = np.stack([np.sum(np.delete(F, i, axis=1) ** 2, axis=1) for i in range(M)], axis=1)
    choosed += list(np.argmin(l2, axis=0))
    choosed = np.asarray(choosed)
    extreme = np.unique(choosed[np.argmax(F[choosed], axis=0)])
    if len(extreme) < M:
        return F.max(axis=0)
    try:
        h = np.linalg.solve(F[extreme], np.ones(M))
        return 1.0 / h
    except np.linalg.LinAlgError:
        return F.max(axis=0)


class IDBEA(LoopAlgorithm):
    """Improved decomposition-based EA: steady-state replacement by (perpendicular, projected) distance to
    reference directions with an adaptive constraint-violation threshold."""

    def initial_size(self):
        W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        self.z = F.min(axis=0)
        self.a = _intercepts(F)

    def step(self):
        rng, N = self.rng, self.N
        for i in range(N):
            partner = int(rng.integers(0, N))
            off = self.evaluate(ga_half(self.problem, decs(self.pop[[i, partner]]), rng=rng))
            fo = objs(off)[0]
            cvp_all = cv(self.pop)
            feasible = cvp_all <= 0
            F = objs(self.pop)
            if not feasible.any() or not np.any(np.all(F[feasible] <= fo, axis=1)):
                lst = rng.permutation(N)
                with np.errstate(all="ignore"):
                    nP = (F[lst] - self.z) / (self.a - self.z)
                    nO = (fo - self.z) / (self.a - self.z)
                normP, normO = np.linalg.norm(nP, axis=1), np.linalg.norm(nO)
                cosP = np.sum(nP * self.W[lst], axis=1) / normP
                cosO = np.sum(nO * self.W[lst], axis=1) / normO
                d1_old, d1_new = normP * cosP, normO * cosO
                d2_old, d2_new = normP * np.sqrt(np.maximum(0, 1 - cosP ** 2)), normO * np.sqrt(np.maximum(0, 1 - cosO ** 2))
                cvo = float(cv(off)[0])
                cvp = cvp_all[lst]
                tau = cvp.mean() * np.sum(cvp == 0) / len(cvp)
                replace = (((d2_new < d2_old) | ((d2_new == d2_old) & (d1_new < d1_old))) &
                           (((cvo < tau) & (cvp < tau)) | (cvo == cvp))) | ((cvo >= tau) & (cvo < cvp))
                hit = np.where(replace)[0]
                if len(hit):
                    self.pop[lst[hit[0]]] = off[0]
                self.a = _intercepts(objs(self.pop))
                self.z = np.minimum(self.z, fo)
