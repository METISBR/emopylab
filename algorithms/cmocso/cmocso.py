# emopylab 2026
"""CMOCSO (competitive and cooperative swarm optimization constrained multi-objective optimization algorithm).

Reference:
F. Ming, W. Gong, D. Li, L. Wang, and L. Gao. A competitive and cooperative swarm optimizer for
constrained multi-objective optimization problems. IEEE Transactions on Evolutionary Computation,
2023, 27(5): 1313-1326.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, adds, cons, decs, ga_half, objs, tournament
from algorithms.community_utils.spea import cal_fitness, overall_cv, select, truncation
from core.population import Population

ALGORITHM_FLAGS = {'CMOCSO': {'constrained', 'large', 'multi', 'real'}}


def _update_feasible(pop, N):
    """Feasible solutions only, truncated to ``N`` among the non-dominated ones."""
    C = cons(pop)
    feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
    pop = pop[feas]
    if len(pop) > N:
        F = objs(pop)
        fit = cal_fitness(F, cons(pop), 0)
        nxt = fit < 1
        K = int(nxt.sum()) - N
        if K > 0:
            idx = np.where(nxt)[0]
            nxt[idx[truncation(F[nxt], K)]] = False
        pop = pop[nxt]
    return pop


class CMOCSO(LoopAlgorithm):
    """A competitive swarm (losers learn from winners, epsilon-constrained fitness) coexists with a cooperative
    population that ignores the constraints (genetic operator); a feasible-only archive is the result."""

    def start(self):
        N, pop = self.N, self.pop
        self.CVmax = float(np.max(overall_cv(cons(pop)))) if cons(pop).size else 0.0
        self.epsilon = self.CVmax
        self.comp, _ = select(pop, N, True, self.epsilon)
        self.coop, self.coop_fit = select(pop, N, True, np.inf)
        self.pop = _update_feasible(pop, N)
        self.Tc, self.cp, self.alpha, self.tao, self.y = 0.9 * np.ceil(self.max_FE / N), 2, 0.95, 0.05, 10.0
        self.G = self.max_FE / N

    def _competitive(self, loser, winner, y):
        rng, D = self.rng, self.D
        LX, WX = decs(loser), decs(winner)
        n = len(LX)
        LV, WV = adds(loser, "V", np.zeros((n, D))), adds(winner, "V", np.zeros((n, D)))
        r1, r2 = np.repeat(rng.random((n, 1)), D, axis=1), np.repeat(rng.random((n, 1)), D, axis=1)
        off_v = r1 * LV + r2 * (WX - LX) * y
        sign = (-1.0) ** int(rng.integers(1, 3))
        off_x = LX + off_v + r1 * (off_v - LV) * sign
        off_x, off_v = np.vstack([off_x, WX]), np.vstack([off_v, WV])
        lower, upper = np.tile(self.lower, (2 * n, 1)), np.tile(self.upper, (2 * n, 1))
        site, mu = rng.random((2 * n, D)) < 1 / D, rng.random((2 * n, D))
        off_x = np.maximum(np.minimum(off_x, upper), lower)
        span = upper - lower
        with np.errstate(all="ignore"):
            t = site & (mu <= 0.5)
            off_x[t] += span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off_x[t] - lower[t]) / span[t]) ** 21) ** (1 / 21) - 1)
            t = site & (mu > 0.5)
            off_x[t] += span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (upper[t] - off_x[t]) / span[t]) ** 21) ** (1 / 21))
        return self.evaluate(off_x, V=off_v)

    def _cooperative(self, parents):
        from algorithms.community_utils.base import ga
        return self.evaluate(ga(self.problem, decs(parents), [1, 20, 1, 20], rng=self.rng))

    def step(self):
        rng, N, M = self.rng, self.N, self.M
        gen = int(np.ceil(self.FE / N))
        CV = overall_cv(cons(self.comp))
        self.CVmax = max(float(CV.max()), self.CVmax)
        eps0 = self.CVmax
        rf = np.sum(CV <= 1e-6) / len(self.comp)
        if gen > self.Tc:
            self.epsilon = 0.0
        elif rf < self.alpha:
            self.epsilon = (1 - self.tao) * self.epsilon
        else:
            self.epsilon = eps0 * ((1 - gen / self.Tc) ** self.cp)
        fit = cal_fitness(objs(self.comp), cons(self.comp), self.epsilon)
        n = len(self.comp)
        rank = rng.permutation(n)[: (n // 2) * 2] if n >= 2 else np.array([0, 0])
        h = len(rank) // 2
        loser, winner = rank[:h].copy(), rank[h:].copy()
        change = fit[loser] <= fit[winner]
        loser[change], winner[change] = winner[change].copy(), loser[change].copy()
        off1 = self._competitive(self.comp[loser], self.comp[winner], self.y)
        pool = tournament(2, N, self.coop_fit, rng=rng)
        off2 = self._cooperative(self.coop[pool])
        off = Population.merge(off1, off2)
        self.pop = _update_feasible(Population.merge(self.pop, off), N)
        self.comp, _ = select(Population.merge(self.comp, off), N, True, self.epsilon)
        gen = int(np.ceil(self.FE / N))
        self.y = M ** 2 * ((gen / self.G) - 1) ** 2 + 1
        self.coop, self.coop_fit = select(Population.merge(off, self.coop), N, True, np.inf)
