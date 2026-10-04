# emopylab 2026
"""IM-C-MOEA-D (inverse modeling constrained MOEA/D).

Reference:
L. R. C. Farias and A. F. R. Araujo. An inverse modeling constrained multi-objective evolutionary
algorithm based on decomposition. Proceedings of the IEEE International Conference on Systems, Mans
and Cybernetics, 2024.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, cons, cv, decs, kmeans, neighbors_of, objs, polynomial_mutation,
                                            tournament, uniform_point)
from algorithms.community_utils.surrogates import gp_linear_predict
from core.population import Population

ALGORITHM_FLAGS = {'IMCMOEAD': {'constrained', 'integer', 'large', 'multi', 'real'}}


def _constraint_objs(pop, M):
    F = objs(pop).copy()
    C = np.maximum(0, cons(pop))
    if C.size:
        bad = np.any(C > 0, axis=1)
        F[bad] = F.max(axis=0) + C[bad].sum(axis=1, keepdims=True) * np.ones((1, M))
    return F


class IMCMOEAD(LoopAlgorithm):
    """Inverse-modeling MOEA/D for constrained problems: offspring come from per-objective linear Gaussian
    process models fitted inside every objective-space cluster."""

    def __init__(self, pop_size: int = 100, K: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.K = int(K)

    def initial_size(self):
        self.T = int(np.ceil(self.pop_size / 10))
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.ideal = objs(self.pop).min(axis=0)

    def _operator(self, parents, L=3):
        rng, M = self.rng, self.M
        X, F = decs(parents), objs(parents)
        N, D = X.shape
        if D < 3:
            L = D
        if len(parents) < 2 * M:
            off = X.copy()
        else:
            blocks = []
            fmin = 1.5 * F.min(axis=0) - 0.5 * F.max(axis=0)
            fmax = 1.5 * F.max(axis=0) - 0.5 * F.min(axis=0)
            for m in range(M):
                sel = rng.permutation(N)[: N // M]
                offd = X[sel].copy()
                for d in rng.permutation(D)[:L]:
                    try:
                        mu, s2 = gp_linear_predict(F[sel, m], X[sel, d], np.linspace(fmin[m], fmax[m], len(offd)))
                        offd[:, d] = mu + rng.random() * np.sqrt(s2) * rng.standard_normal(len(s2))
                    except np.linalg.LinAlgError:
                        pass
                blocks.append(offd)
            off = np.vstack(blocks)
        lo, up = self.lower, self.upper
        rand = lo + rng.random(off.shape) * (up - lo)
        bad = (off < lo) | (off > up)
        off[bad] = rand[bad]
        return self.evaluate(polynomial_mutation(off, lo, up, rng))

    def step(self):
        rng, W, B, T, pop, N = self.rng, self.W, self.B, self.T, self.pop, self.N
        part = kmeans(objs(pop), self.K, rng)
        offs = []
        for k in np.unique(part):
            parents = pop[part == k]
            pool = tournament(2, len(parents), cv(parents), rng=rng)
            offs.append(self._operator(parents[pool]))
        off = Population.merge(*offs) if len(offs) > 1 else offs[0]
        PF, OF = _constraint_objs(pop, self.M), _constraint_objs(off, self.M)
        self.ideal = np.minimum(self.ideal, OF.min(axis=0))
        nadir = np.vstack([PF, OF]).max(axis=0)
        with np.errstate(all="ignore"):
            PF = (PF - self.ideal) / (nadir - self.ideal)
            OF = (OF - self.ideal) / (nadir - self.ideal)
        cvo_all = cv(off)
        for i in range(len(off)):
            tch = np.max(OF[i] * W, axis=1)
            best = int(np.argmin(tch))
            P = B[best][rng.permutation(B.shape[1])]
            cvp = cv(pop[P])
            g_old = np.max(PF[P] * W[P], axis=1)
            g_new = np.max(OF[i] * W[P], axis=1)
            for j in np.where((g_old >= g_new) & (cvp >= cvo_all[i]))[0][:T]:
                pop[P[j]] = off[i]
        self.pop = pop
