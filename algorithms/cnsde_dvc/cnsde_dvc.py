# emopylab 2026
"""CNSDE-DVC (constrained nondominated sorting differential evolution based on decision variable classification).

Reference:
W. Du, W. Zhong, Y. Tang, W. Du, and Y. Jin. High-dimensional robust multi-objective optimization
for order scheduling: A decision variable classification approach. IEEE Transactions on Industrial
Informatics, 2019, 15(1): 293-304.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, de, decs, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'CNSDEDVC': {'integer', 'multi', 'real', 'robust'}}


class CNSDEDVC(LoopAlgorithm):
    """Non-dominated sorting DE with decision-variable classification: variables are split into highly and weakly
    robustness-related by probing their effect on the objectives, then evolved with robust / plain selection."""

    def __init__(self, pop_size: int = 100, sampling=None, SN=4, PN=6, TN=15, theta=0.001, eta=0.001, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.SN, self.PN, self.TN, self.theta, self.eta = int(SN), int(PN), int(TN), float(theta), float(eta)

    def _dvc(self):
        pop, rng, D, N = self.pop, self.rng, self.D, len(self.pop)
        lo, up = self.lower, self.upper
        X, F = decs(pop), objs(pop)
        delta_rel = float(getattr(self.problem, "delta", 0.1))
        allval = np.zeros((self.TN, D))
        for T in range(self.TN):
            var = np.zeros((self.SN, D))
            for i in range(D):
                a = rng.permutation(N)[: self.SN]
                delta = delta_rel * (up[i] - lo[i])
                P = np.repeat(X[a], self.PN, axis=0)
                P[:, i] += 2 * delta * rng.random(len(P)) - delta
                new = objs(self.evaluate(P)).reshape(self.SN, self.PN, -1)
                vcv = np.sum(np.abs(new - F[a][:, None, :]), axis=2)
                var[:, i] = np.var(vcv, axis=1, ddof=1)
            allval[T] = var.mean(axis=0)
        tval = np.sum(allval < self.theta, axis=0).astype(float)
        m = tval.mean()
        self.HR, self.LR = np.where(tval <= m)[0], np.where(tval > m)[0]

    def _select(self, pop, robust):
        N, F = self.N, objs(pop)
        if robust:
            X = decs(pop)
            Fp, _ = self.problem.perturb(X, rng=self.rng)[:2]
            dr = np.sum(np.abs(Fp.mean(axis=0) - F), axis=1)
            dr[dr <= self.eta] = 0
            front, maxf = nd_sort(F, dr[:, None], N)
        else:
            front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        cd = crowding(F, front)
        last = np.where(front == maxf)[0]
        nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
        return pop[nxt], front[nxt], cd[nxt]

    def start(self):
        self._dvc()
        self.pop, self.front, self.crowd = self._select(self.pop, False)

    def _subgen(self, idx, robust):
        rng, pop = self.rng, self.pop
        n = len(pop)
        X = decs(pop)
        off = X[tournament(2, n, self.front, -self.crowd, rng=rng)].copy()
        new = de(self.problem, X, X[rng.integers(0, n, n)], X[rng.integers(0, n, n)], [0.9, 0.5, 1, 20], rng=rng)
        off[:, idx] = new[:, idx]
        self.pop, self.front, self.crowd = self._select(Population.merge(pop, self.evaluate(off)), robust)

    def step(self):
        if len(self.HR):
            for _ in range(10):
                self._subgen(self.HR, True)
        if len(self.LR):
            for _ in range(2):
                self._subgen(self.LR, False)
