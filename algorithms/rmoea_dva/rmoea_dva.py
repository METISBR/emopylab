# emopylab 2026
"""RMOEA-DVA (robust multi-objective evolutionary algorithm with decision variable assortment).

Reference:
J. Liu, Y. Liu, Y. Jin, and F. Li. A decision variable assortment-based evolutionary algorithm for
dominance robust multiobjective optimization. IEEE Transactions on Systems, Man, and Cybernetics:
Systems, 2022, 52(5): 3360-3375.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga, nd_sort, objs, tournament
from algorithms.community_utils.robust import perturbed_objs

ALGORITHM_FLAGS = {'RMOEADVA': {'integer', 'multi', 'real', 'robust'}}


class RMOEADVA(LoopAlgorithm):
    """Decision variables are split by the sensitivity of the Pareto rank to disturbances of each variable
    (DVA): low-robustness-related variables evolve with plain NSGA-II selection, high ones with a selection that
    adds a dominance-robustness index computed from disturbed copies of every solution."""

    def __init__(self, pop_size: int = 100, n_dva: int = 50, theta: float = 0.3, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.n_dva, self.theta = int(n_dva), float(theta)

    def _dva(self, pop):
        rng, X = self.rng, decs(pop)
        N, D = X.shape
        n_dva = min(self.n_dva, N)  # the reference draws without repetition, so it needs n_dva <= N
        H = int(getattr(self.problem, "H", 50))
        delta_rel = float(getattr(self.problem, "delta", 0.05))
        sd = np.zeros((n_dva, D))
        for i in range(D):
            base = X[rng.permutation(N)[:n_dva]]
            delta = delta_rel * (self.upper[i] - self.lower[i])
            for j in range(n_dva):
                P = np.repeat(base[[j]], H, axis=0)
                P[:, i] += 2 * delta * rng.random(H) - delta
                front, _ = nd_sort(objs(self.evaluate(P)), None, np.inf)
                sd[j, i] = np.std(front, ddof=1) if H > 1 else 0.0
        span = sd.max() - sd.min()
        sd = (sd - sd.min()) / span if span > 0 else np.zeros_like(sd)  # constant sensitivities: all variables low
        m = sd.mean(axis=0)
        self.HR, self.LR = np.where(m > self.theta)[0], np.where(m <= self.theta)[0]

    def _select(self, pop, N, robust):
        F = objs(pop)
        if robust:
            X = decs(pop)
            Fp = perturbed_objs(self.problem, X, rng=self.rng)          # (H, n, M)
            n = len(pop)
            dri = np.zeros(n)
            for i in range(n):
                front, maxf = nd_sort(Fp[:, i, :], None, np.inf)
                dri[i] = sum(np.sum(front == j) * (j - 1) for j in range(1, int(maxf) + 1)) / Fp.shape[0]
            span = dri.max() - dri.min()
            dri = (dri - dri.min()) / span if span > 0 else np.zeros(n)
            front, maxf = nd_sort(F + (1.0 / (1.0 - dri + 1e-6))[:, None], None, N)
            nxt = front < maxf
            last = np.where(front == maxf)[0]
            rank = np.argsort(dri[last], kind="stable")
            nxt[last[rank[: N - int(nxt.sum())]]] = True
            return pop[nxt], front[nxt], dri[nxt]
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        cd = crowding(F, front)
        last = np.where(front == maxf)[0]
        rank = np.argsort(-cd[last], kind="stable")
        nxt[last[rank[: N - int(nxt.sum())]]] = True
        return pop[nxt], front[nxt], cd[nxt]

    def start(self):
        self._dva(self.pop)
        if len(self.HR) == 0 and len(self.LR) == 0:  # undefined sensitivities: treat every variable as low
            self.LR = np.arange(self.D)
        _, self.front, self.crowd = self._select(self.pop, self.N, False)

    def _subgen(self, idx, robust):
        pop, rng, N = self.pop, self.rng, self.N
        X = decs(pop)
        n = len(pop)
        Off = X[tournament(2, n, self.front, -self.crowd, rng=rng)]
        New = ga(self.problem, X[rng.integers(0, n, n)], rng=rng)
        Off[:, idx] = New[:, idx]
        off = self.evaluate(Off)
        from core.population import Population
        self.pop, self.front, self.crowd = self._select(Population.merge(pop, off), N, robust)

    def step(self):
        if len(self.LR):
            self._subgen(self.LR, False)
        if len(self.HR):
            self._subgen(self.HR, True)
