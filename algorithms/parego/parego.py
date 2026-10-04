# emopylab 2026
"""ParEGO (efficient global optimization for Pareto optimization).

Reference:
J. Knowles. ParEGO: A hybrid algorithm with on-line landscape approximation for expensive
multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2006, 10(1):
50-66.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, objs, tournament, uniform_point
from algorithms.community_utils.dace import DaceModel, norm_cdf, norm_pdf
from core.population import Population

ALGORITHM_FLAGS = {'ParEGO': {'expensive', 'integer', 'multi', 'real'}}


class ParEGO(LoopAlgorithm):
    """Each iteration scalarises the (normalised) objectives with a random weight vector, fits a Kriging model to the best
    solutions and evaluates the point of maximum expected improvement found by a small GA on the surrogate."""

    def __init__(self, pop_size: int = 100, ifes: int = 10000, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.ifes = int(ifes)

    def _initialize_infill(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        from operators.utility_functions.UniformPoint import UniformPoint
        n = 11 * self.D - 1
        P, _ = UniformPoint(n, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.theta = 10.0 * np.ones(self.D)
        self._set_optimum()

    def _evol_alg(self, pcheby, dec, model):
        rng = self.rng
        n0 = len(dec)
        off = np.vstack([ga(self.problem, dec[tournament(2, n0, pcheby, rng=rng)], rng=rng),
                         ga(self.problem, dec, [0, 0, 1, 20], rng=rng)])
        N = len(off)
        gbest = pcheby.min()
        e0, best = np.inf, off[0]
        ifes = self.ifes
        while ifes > 0:
            y, mse = model.predict(off, mse=True)
            with np.errstate(all="ignore"):
                s = np.sqrt(mse)
                z = (gbest - y) / s
                ei = -(gbest - y) * norm_cdf(z) - s * norm_pdf(z)
            order = np.argsort(np.where(np.isnan(ei), np.inf, ei), kind="stable")
            if ei[order[0]] < e0:
                best, e0 = off[order[0]], ei[order[0]]
            parent = off[order[: int(np.ceil(N / 2))]]
            pe = ei[order[: int(np.ceil(N / 2))]]
            off = np.vstack([ga(self.problem, parent[tournament(2, len(parent), pe, rng=rng)], rng=rng),
                             ga(self.problem, parent, [0, 0, 1, 20], rng=rng)])
            ifes -= len(off)
        return best

    def step(self):
        rng, D = self.rng, self.D
        pop = self.pop
        lam = self.W[int(rng.integers(0, len(self.W)))]
        F = objs(pop)
        N = len(F)
        with np.errstate(all="ignore"):
            Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        pch = np.max(Fn * lam, axis=1) + 0.05 * np.sum(Fn * lam, axis=1)
        if N > 11 * D - 1 + 25:
            nxt = np.argsort(pch, kind="stable")[: 11 * D - 1 + 25]
        else:
            nxt = np.arange(N)
        pdec, pch = decs(pop)[nxt], pch[nxt]
        d1 = np.unique(np.round(pdec * 1e6) / 1e6, axis=0, return_index=True)[1]
        d2 = np.unique(np.round(pch * 1e6) / 1e6, return_index=True)[1]
        distinct = np.intersect1d(d1, d2)
        pdec, pch = pdec[distinct], pch[distinct]
        model = DaceModel(pdec, pch, "regpoly1", self.theta, 1e-5 * np.ones(D), 20 * np.ones(D))
        self.theta = model.theta
        x = self._evol_alg(pch, decs(pop), model)
        self.pop = Population.merge(pop, self.evaluate(x[None, :]))
