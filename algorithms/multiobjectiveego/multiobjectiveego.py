# emopylab 2026
"""MultiObjectiveEGO (multi-objective efficient global optimization).

Reference:
R. Hussein and K. Deb. A generative Kriging surrogate model for constrained and unconstrained multi-
objective optimization. Proceedings of the Genetic and Evolutionary Computation Conference, 2016,
573-580.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, objs, tournament, uniform_point
from algorithms.community_utils.dace import DaceModel, norm_cdf, norm_pdf
from core.population import Population

ALGORITHM_FLAGS = {'MultiObjectiveEGO': {'constrained', 'expensive', 'integer', 'multi', 'real'}}


class MultiObjectiveEGO(LoopAlgorithm):
    """A reference direction at a time: the solutions closest to the direction are scalarised by an S-metric (Tchebycheff plus
    a small sum term, infeasible ones scored by their scaled violation), a Kriging model of that scalar is built and the point
    with maximal expected improvement, found by a real-coded GA, is evaluated; ``num_k`` points are added per direction."""

    def __init__(self, pop_size: int = 100, alpha: float = 0.7, num_k: int = 5, h: int = 21, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha, self.num_k, self.H = float(alpha), int(num_k), int(h)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        P, _ = UniformPoint(self.pop_size, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        R, _ = uniform_point(self.H, self.M)
        self.R = R / np.linalg.norm(R, axis=1, keepdims=True)
        self.direction, self.k = 0, 0
        self._set_optimum()

    def _rga(self, fun):
        rng, D = self.rng, self.D
        from operators.utility_functions.UniformPoint import UniformPoint
        n = 10 * D
        P, _ = UniformPoint(n, D, "Latin", rng=rng)
        off = self.lower + np.asarray(P) * (self.upper - self.lower)
        best, best_v = off[0], np.inf
        for _ in range(100):
            v = fun(off)
            order = np.argsort(v, kind="stable")
            if v[order[0]] < best_v:
                best, best_v = off[order[0]], v[order[0]]
            keep = order[: int(np.ceil(n / 2))]
            parent = off[keep]
            off = np.vstack([ga(self.problem, parent[tournament(2, len(parent), v[keep], rng=rng)], rng=rng),
                             ga(self.problem, parent, [0.9, 2, 1.0 / D, 20], rng=rng)])
        return best

    def step(self):
        D, rho = self.D, 1e-3
        self.not_terminated(self.pop)
        pop = self.pop
        Xt, Ft, Ct = decs(pop), objs(pop), cons(pop)
        r = self.R[self.direction]
        normP = np.linalg.norm(Ft, axis=1)
        with np.errstate(all="ignore"):
            cosine = np.sum(Ft * r, axis=1) / normP / np.linalg.norm(r)
            dist = normP * np.sqrt(np.maximum(1 - cosine ** 2, 0))
        n = int(np.ceil(self.alpha * len(pop)))
        idx = np.argsort(np.where(np.isnan(dist), np.inf, dist), kind="stable")[:n]
        F, X = Ft[idx], Xt[idx]
        C = Ct[idx] if Ct.size else np.zeros((n, 0))
        viol = np.maximum(C, 0)                                        # violation of g <= 0
        feas = np.all(viol == 0, axis=1) if C.size else np.ones(n, bool)
        with np.errstate(all="ignore"):
            Fs = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
            S = np.zeros(n)
            tmp = Fs[feas] / r
            S[feas] = np.max(tmp, axis=1) + rho * np.sum(tmp, axis=1)
            if (~feas).any():
                rng_c = viol.max(axis=0) - viol.min(axis=0)
                Cs = (viol - viol.min(axis=0)) / np.where(rng_c == 0, 1.0, rng_c)
                asf_max = S[feas].max() if feas.any() else 0.0
                S[~feas] = asf_max + np.sum(Cs[~feas], axis=1)
        S = np.nan_to_num(S)
        keep = np.unique(np.round(X * 1e10) / 1e10, axis=0, return_index=True)[1]      # Kriging needs distinct sites
        model = DaceModel(X[keep], S[keep], "regpoly0", np.ones(D), 0.001 * np.ones(D), 1000 * np.ones(D))
        f_min = S.min()

        def neg_ei(x):
            y, mse = model.predict(x, mse=True)
            s = np.sqrt(np.maximum(0, mse))
            with np.errstate(all="ignore"):
                z = (f_min - y) / s
                ei = (f_min - y) * norm_cdf(z) + s * norm_pdf(z)
            return np.where(np.isnan(ei), 0.0, -ei)

        best = self._rga(neg_ei)
        if np.min(np.linalg.norm(Xt - best, axis=1)) < 1e-8:
            best = self._rga(lambda x: -np.min(np.sqrt(np.maximum(np.sum(x * x, 1)[:, None] + np.sum(Xt * Xt, 1)[None, :] - 2 * x @ Xt.T, 0)), axis=1))
        self.pop = Population.merge(pop, self.evaluate(best[None, :]))
        self.k += 1
        if self.k >= self.num_k:
            self.k = 0
            self.direction = (self.direction + 1) % len(self.R)
