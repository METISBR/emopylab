# emopylab 2026
"""MONAS (multi-objective neural architecture search).

Reference:
F. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin. An evolutionary framework for multi-objective neural
architecture search. IEEE Transactions on Evolutionary Computation, 2025.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, ga, objs, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'MONAS': {'binary', 'integer', 'label', 'multi', 'multimodal', 'permutation', 'real'}}


def _pdist(A):
    return np.sqrt(np.maximum(np.sum(A * A, 1)[:, None] + np.sum(A * A, 1)[None, :] - 2.0 * A @ A.T, 0.0))


def _fitness(F, X, density_space):
    """Strength/raw fitness from objective dominance plus a k-th nearest neighbour density measured in the objective
    (``'obj'``) or decision (``'dec'``) space; returns ``(density, fitness)``."""
    N = len(F)
    dom = ((F[:, None, :] < F[None, :, :]).any(axis=2) & ~(F[:, None, :] > F[None, :, :]).any(axis=2))
    S = dom.sum(axis=1)
    R = S @ dom
    dist = _pdist(F if density_space == "obj" else X)
    np.fill_diagonal(dist, np.inf)
    dist = np.sort(dist, axis=1)
    D = 1.0 / (dist[:, max(int(np.floor(np.sqrt(N))) - 1, 0)] + 2)
    return D, R + D


def _selection(pop, N, space):
    F, X = objs(pop), decs(pop)
    D, fit = _fitness(F, X, space)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        dist = _pdist(F[nxt] if space == "obj" else X[nxt])
        np.fill_diagonal(dist, np.inf)
        nxt[idx[truncate_lexi(dist, int(nxt.sum()) - N)]] = False
    pop, fit, D = pop[nxt], fit[nxt], D[nxt]
    o = np.argsort(fit, kind="stable")
    return pop[o], fit[o], D[o]


class MONAS(LoopAlgorithm):
    """Two populations evolve on the same offspring: the first is selected by objective-space diversity, the second by
    decision-space diversity (so distinct Pareto-optimal sets are kept); the first half of the run uses SBX/polynomial
    mutation, the second half differential evolution.  The result is the objective-space population."""

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(self.pop_size))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop1 = infills
        self.pop2 = self.evaluate(self.random_decs(self.pop_size))
        self.d_dec, self.fit1 = _fitness(objs(self.pop1), decs(self.pop1), "obj")
        self.d, self.fit2 = _fitness(objs(self.pop2), decs(self.pop2), "dec")
        self.pop = self.pop1
        self._set_optimum()

    def step(self):
        rng, N = self.rng, self.N
        p1, p2 = self.pop1, self.pop2
        if self.FE <= self.max_FE / 2:
            m1 = tournament(2, N, self.d_dec, self.fit1, rng=rng)
            m2 = tournament(2, N, self.d, self.fit2, rng=rng)
            o1 = self.evaluate(ga(self.problem, decs(p1[m1]), rng=rng))
            o2 = self.evaluate(ga(self.problem, decs(p2[m2]), rng=rng))
        else:
            m1 = tournament(2, 2 * N, self.d_dec, self.fit1, rng=rng)
            m2 = tournament(2, 2 * N, self.d, self.fit2, rng=rng)
            X1, X2 = decs(p1), decs(p2)
            o1 = self.evaluate(de(self.problem, X1, X1[m1[: len(m1) // 2]], X1[m1[len(m1) // 2:]], rng=rng))
            o2 = self.evaluate(de(self.problem, X2, X2[m2[: len(m2) // 2]], X2[m2[len(m2) // 2:]], rng=rng))
        off = Population.merge(o1, o2)
        self.pop1, self.fit1, self.d_dec = _selection(Population.merge(p1, off), N, "obj")
        self.pop2, self.fit2, self.d = _selection(Population.merge(p2, off), N, "dec")
        self.pop = self.pop1
