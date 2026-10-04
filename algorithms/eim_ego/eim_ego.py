# emopylab 2026
"""EIM-EGO (expected improvement matrix based efficient global optimization).

Reference:
D. Zhan, Y. Cheng, and J. Liu. Expected improvement matrix-based infill criteria for expensive
multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2017, 21(6): 956-975.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, tournament
from algorithms.community_utils.dace import DaceModel, norm_cdf, norm_pdf
from core.population import Population

ALGORITHM_FLAGS = {'EIMEGO': {'expensive', 'integer', 'multi', 'real'}}


class EIMEGO(LoopAlgorithm):
    """One Kriging model per (scaled) objective; the infill point maximises an aggregate of the per-objective expected
    improvements over every current non-dominated point (Euclidean, maximin or hypervolume-style aggregation)."""

    def __init__(self, pop_size: int = 100, criterion: int = 1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.criterion = int(criterion)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        n = 11 * self.D - 1
        P, _ = UniformPoint(n, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _pick_point(self, models, F):
        rng, D, M = self.rng, self.D, self.M
        f = F[nd_sort(F, None, 1)[0] == 1]
        n_ga = 100
        from operators.utility_functions.UniformPoint import UniformPoint
        P, _ = UniformPoint(n_ga, D, "Latin", rng=rng)
        off = self.lower + np.asarray(P) * (self.upper - self.lower)
        best, e_max = off[0], np.inf
        for _ in range(100):
            preds = [m.predict(off, mse=True) for m in models]
            u = np.column_stack([p[0] for p in preds])
            s = np.sqrt(np.maximum(0, np.column_stack([p[1] for p in preds])))
            diff = f[:, None, :] - u[None, :, :]                         # (p, pop, M)
            with np.errstate(all="ignore"):
                z = diff / s[None]
                ei = diff * norm_cdf(z) + s[None] * norm_pdf(z)
            if self.criterion == 1:
                eim = -np.min(np.sqrt(np.sum(ei ** 2, axis=2)), axis=0)
            elif self.criterion == 2:
                eim = -np.min(np.max(ei, axis=2), axis=0)
            else:
                eim = -np.min(np.prod(1.1 - f[:, None, :] + ei, axis=2) - np.prod(1.1 - f, axis=1)[:, None], axis=0)
            order = np.argsort(np.where(np.isnan(eim), np.inf, eim), kind="stable")
            if eim[order[0]] < e_max:
                best, e_max = off[order[0]], eim[order[0]]
            keep = order[: int(np.ceil(n_ga / 2))]
            parent = off[keep]
            off = np.vstack([ga(self.problem, parent[tournament(2, len(parent), eim[keep], rng=rng)], rng=rng),
                             ga(self.problem, parent, [0, 0, 1, 20], rng=rng)])
        return best

    def step(self):
        D, M = self.D, self.M
        pop = self.pop
        X, F = decs(pop), objs(pop)
        with np.errstate(all="ignore"):
            Fs = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        models = [DaceModel(X, Fs[:, i], "regpoly0", np.ones(D), 0.001 * np.ones(D), 1000 * np.ones(D)) for i in range(M)]
        x = self._pick_point(models, Fs)
        self.pop = Population.merge(pop, self.evaluate(x[None, :]))
