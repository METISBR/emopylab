# emopylab 2026
"""DVCEA (decision variables classification-based evolutionary algorithm).

Reference:
X. Ban, J. Liang, K. Qiao, K. Yu, Y. Wang, J. Zhu, B. Qu. A decision variables classification-based
evolutionary algorithm for constrained multi-objective optimization problems. IEEE/CAA Journal of
Automatica Sinica, 2025, 12(9): 1830-1849.
"""

from __future__ import annotations

import numpy as np

from algorithms.apsea.apsea import _epsilon_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, kmeans, objs
from algorithms.community_utils.eps_ea import cal_fitness_eps, de_pbest_1, gn_r1r2r3
from algorithms.community_utils.spea import overall_cv
from core.population import Population

ALGORITHM_FLAGS = {'DVCEA': {'constrained', 'integer', 'large', 'many', 'multi', 'real'}}


class DVCEA(LoopAlgorithm):
    """Variables are classified by how much perturbing them changes the constraint violation (probes around five cluster
    centres): constraint-related variables evolve with a pbest DE, the others with a paired best/worst DE, both under an
    epsilon-relaxed constraint handling whose threshold decays over the run."""

    def start(self):
        pop, N, D, rng = self.pop, self.N, self.D, self.rng
        lab = kmeans(decs(pop), 5, rng)
        C = np.vstack([decs(pop)[lab == k].mean(axis=0) for k in range(5)])
        PN, SN = 4, len(C)
        var_con = np.zeros((SN, D))
        for d in range(D):
            per = np.linspace(self.lower[d], self.upper[d], PN)
            for j in range(SN):
                X = np.tile(C[j], (PN, 1))
                X[:, d] = per
                probe = self.evaluate(X)
                var_con[j, d] = np.std(overall_cv(cons(probe)), ddof=1)
        mean_con = var_con.mean(axis=0)
        self.fea, self.infea = np.where(mean_con > 1e-5)[0], np.where(mean_con <= 1e-5)[0]
        e0 = float(overall_cv(cons(pop)).max()) if cons(pop).size else 0.0
        self.eps0 = e0 if e0 != 0 else 1.0
        self.fit = cal_fitness_eps(objs(pop), cons(pop) if cons(pop).size else None, self.eps0)

    def _pbest_offspring(self, pop, fit, fea, p=0.1):
        rng, N = self.rng, self.N
        X = decs(pop)
        perm = rng.permutation(N) + 1
        r1, r2, _ = gn_r1r2r3(rng, N, perm)
        arr = perm - 1
        best = np.argsort(fit, kind="stable")
        pnp = max(int(np.round(p * N)), 2)
        ri = np.maximum(1, np.ceil(rng.random(N) * pnp).astype(int))
        pbest = X[best[ri - 1]]
        new = de_pbest_1(rng, self.lower, self.upper, X[arr], pbest, X[r1[:N] - 1], X[r2[:N] - 1])
        off = X.copy()
        off[:, fea] = new[:, fea]
        return self.evaluate(off)

    def _better_offspring(self, pop, infea, epsilon):
        rng, D = self.rng, self.D
        X = decs(pop).copy()
        fit1 = cal_fitness_eps(objs(pop), cons(pop) if cons(pop).size else None, epsilon)
        out = []
        while len(X) > 1:
            F = (0.6, 0.8, 1.0)[int(np.searchsorted([1 / 3, 2 / 3], rng.random()))]
            CR = (0.1, 0.2, 1.0)[int(np.searchsorted([1 / 3, 2 / 3], rng.random()))]
            n = len(X)
            idx = list(range(n))
            xr1 = idx.pop(int(np.floor(rng.random() * (n - 1))))
            xr2 = idx[int(np.floor(rng.random() * (n - 2)))]
            # the reference keeps the original fitness vector while the population shrinks
            f1, f2 = fit1[xr1] if xr1 < len(fit1) else np.inf, fit1[xr2] if xr2 < len(fit1) else np.inf
            best, worst = (xr1, xr2) if f1 < f2 else (xr2, xr1)
            p = X.copy()
            v = p[best] + F * (p[best] - p[worst])
            take = rng.random(D) <= 0.5
            p[worst, take] = p[best, take]
            v2 = p[worst].copy()
            t = rng.random(D) < CR
            t[int(np.floor(rng.random() * D))] = True
            v = np.where(t, v, p[best])
            v1 = p[best].copy()
            v1[infea] = v[infea]
            w2 = p[worst].copy()
            w2[infea] = v2[infea]
            out += [v1, w2]
            X = np.delete(X, [xr1, xr2], axis=0)
        out = np.array(out)
        from algorithms.community_utils.base import polynomial_mutation
        out = polynomial_mutation(out, self.lower, self.upper, rng, 20.0, prob=1.0 / D)
        return self.evaluate(out)

    def step(self):
        N = self.N
        pop = self.pop
        cp = (-np.log(self.eps0) - 6) / np.log(1 - 0.5)
        eps = self.eps0 * (1 - self.FE / self.max_FE) ** cp
        off = self._pbest_offspring(pop, self.fit, self.fea)
        pop, _ = _epsilon_selection(Population.merge(pop, off), N, eps)
        off = self._better_offspring(pop, self.infea, eps)
        self.pop, self.fit = _epsilon_selection(Population.merge(pop, off), N, eps)
