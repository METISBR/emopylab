# emopylab 2026
"""PB-RVEA (rVEA based on Pareto based bi-indicator infill sampling criterion).

Reference:
Z. Song, H. Wang, and H. Xu. A framework for expensive many-objective optimization with Pareto-based
bi-indicator infill sampling criterion. Memetic Computing, 2022, 14: 179-191.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, uniform_point
from algorithms.community_utils.dace import DaceModel
from algorithms.k_rvea.k_rvea import _angles, _env_selection, _update_archive
from core.population import Population

ALGORITHM_FLAGS = {'PBRVEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


class PBRVEA(LoopAlgorithm):
    """Reference-vector guided search on Kriging surrogates. After ``wmax`` surrogate-only RVEA generations, the survivors are
    ranked on two indicators against every solution evaluated so far: convergence (distance to the ideal point) and diversity
    (distance to the nearest evaluated solution); the whole first Pareto front of these two indicators is evaluated for real."""

    def __init__(self, pop_size: int = 100, alpha: float = 2, wmax: int = 15, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha, self.wmax = float(alpha), int(wmax)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.V0, self.pop_size = uniform_point(self.pop_size, self.M)
        self.V = self.V0.copy()
        self.V1 = self.V0.copy()
        self.NI = self.N
        P, _ = UniformPoint(self.NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills          # archive used for the search population
        self.A = infills            # every solution evaluated so far
        self.theta = 5.0 * np.ones((self.M, self.D))
        self._set_optimum()

    def step(self):
        rng, D, M = self.rng, self.D, self.M
        Dec, Obj = decs(self.pop), objs(self.pop)
        train_X, train_Y = decs(self.A), objs(self.A)
        distinct = np.unique(np.round(train_X * 1e6) / 1e6, axis=0, return_index=True)[1]
        train_X, train_Y = train_X[distinct], train_Y[distinct]
        models = []
        for i in range(M):
            dm = DaceModel(train_X, train_Y[:, i], "regpoly0", self.theta[i], 1e-5 * np.ones(D), 100 * np.ones(D))
            models.append(dm)
            self.theta[i] = dm.theta
        V, V0 = self.V, self.V0
        for w in range(1, self.wmax + 1):
            off = ga(self.problem, Dec, rng=rng)
            off_obj = np.column_stack([m.predict(off) for m in models])
            pop_obj, pop_dec = np.vstack([Obj, off_obj]), np.vstack([Dec, off])
            index = _env_selection(pop_obj, V, (w / self.wmax) ** self.alpha)
            Dec, Obj = pop_dec[index], pop_obj[index]
            if w % int(np.ceil(self.wmax * 0.1)) == 0:
                V = V.copy()
                V[: len(V0)] = V0 * (Obj.max(axis=0) - Obj.min(axis=0))
        DA_obj = objs(self.pop)
        allo = np.vstack([Obj, DA_obj])
        lo, hi = allo.min(axis=0), allo.max(axis=0)
        with np.errstate(all="ignore"):
            DA_nor = (DA_obj - lo) / (hi - lo)
            pre = (Obj - lo) / (hi - lo)
        zmin = np.vstack([DA_nor, pre]).min(axis=0)
        dist_D = np.sqrt(((pre[:, None, :] - DA_nor[None, :, :]) ** 2).sum(axis=2))
        DI = -dist_D.min(axis=1)
        CI = np.sqrt(((pre - zmin) ** 2).sum(axis=1))
        front, _ = nd_sort(np.column_stack([DI, CI]), None, 1)
        new_dec = np.unique(Dec[front == 1], axis=0)
        new = self.evaluate(new_dec)
        self.A = Population.merge(self.A, new)
        self.pop = _update_archive(rng, self.pop, new, self.V1, len(new_dec), self.NI)
        adapt = self.V0 * (objs(self.pop).max(axis=0) - objs(self.pop).min(axis=0))
        self.V1 = self.V1.copy()
        self.V1[: len(V0)] = adapt
        self.V = self.V1
