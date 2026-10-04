# emopylab 2026
"""PB-NSGA-III (nSGA-III based on Pareto based bi-indicator infill sampling criterion).

Reference:
Z. Song, H. Wang, and H. Xu. A framework for expensive many-objective optimization with Pareto-based
bi-indicator infill sampling criterion. Memetic Computing, 2022, 14: 179-191.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import nsga3_ref
from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, uniform_point
from algorithms.community_utils.bi_indicator import bi_indicator_select, train_models
from core.population import Population

ALGORITHM_FLAGS = {'PBNSGAIII': {'expensive', 'integer', 'many', 'multi', 'real'}}


class PBNSGAIII(LoopAlgorithm):
    """NSGA-III evolved for ``wmax`` generations on Kriging surrogates (reference-point niching on the last front); the first
    Pareto front of the convergence/diversity indicators is then evaluated for real and merged into the archive by NSGA-III
    environmental selection."""

    def __init__(self, pop_size: int = 100, wmax: int = 15, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax = int(wmax)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.NI = self.N
        P, _ = UniformPoint(self.NI, self.D, "Latin", rng=self.rng)
        self.W, _ = uniform_point(self.N, self.M)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.A = infills
        self.zmin = objs(infills).min(axis=0)
        self.theta = 5.0 * np.ones((self.M, self.D))
        self._set_optimum()

    def step(self):
        rng, NI = self.rng, self.NI
        Dec, Obj = decs(self.pop), objs(self.pop)
        models = train_models(self.A, self.theta)
        zmin = self.zmin
        for _ in range(self.wmax):
            off = ga(self.problem, Dec[rng.integers(0, len(Dec), NI)], rng=rng)
            off_obj = np.column_stack([m.predict(off) for m in models])
            zmin = np.minimum(zmin, off_obj.min(axis=0))
            all_obj, all_dec = np.vstack([Obj, off_obj]), np.vstack([Dec, off])
            front, maxf = nd_sort(all_obj, None, NI)
            nxt = front < maxf
            last = np.where(front == maxf)[0]
            choose = nsga3_ref.last_selection(all_obj[nxt], all_obj[last], NI - int(nxt.sum()), self.W, zmin, rng)
            nxt[last[choose]] = True
            Dec, Obj = all_dec[nxt], all_obj[nxt]
        new = self.evaluate(bi_indicator_select(Dec, Obj, objs(self.pop)))
        self.A = Population.merge(self.A, new)
        merged = Population.merge(self.pop, new)
        self.zmin = objs(merged).min(axis=0)
        self.pop = nsga3_ref.select(merged, self.N, self.W, self.zmin, rng)
