# emopylab 2026
"""EMMOEA (expensive multi-/many-objective evolutionary algorithm).

Reference:
S. Qin, C. Sun, Q. Liu, and Y. Jin. A performance indicator-based infill criterion for expensive
multi-/many-objective optimization. IEEE Transactions on Evolutionary Computation, 2023, 27(4):
1085-1099.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, uniform_point
from algorithms.community_utils.dace import DaceModel, norm_cdf, norm_pdf
from algorithms.k_rvea.k_rvea import _angles, _argmin_rows
from core.population import Population

ALGORITHM_FLAGS = {'EMMOEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


def _norm01(F):
    with np.errstate(all="ignore"):
        return (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))


def kriging_selection(F, V):
    """Closest-to-ideal solution (normalised objectives) of every active reference vector."""
    F = _norm01(F)
    assoc = _argmin_rows(_angles(F, V))
    nxt = []
    for i in np.unique(assoc):
        cur = np.where(assoc == i)[0]
        nxt.append(cur[int(np.argmin(np.sqrt((F[cur] ** 2).sum(1))))])
    return np.array(nxt, dtype=int)


class EMMOEA(LoopAlgorithm):
    """Objective Kriging models drive ``gmax`` reference-vector generations; a further Kriging model learns a scalar
    performance indicator (distance to the ideal point minus distance to the nearest evaluated neighbour) and the candidate
    with the largest expected improvement of that indicator is evaluated; the non-dominated representatives of every
    reference vector are re-injected into the search population."""

    def __init__(self, pop_size: int = 100, gmax: int = 10, ni: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.gmax, self.ni = int(gmax), int(ni)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.V, _ = uniform_point(self.N, self.M)
        P, _ = UniformPoint(self.ni, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        D = self.D
        self.lob, self.upb = 1e-5 * np.ones(D), 100 * np.ones(D)
        self.theta, self.theta0 = 5.0 * np.ones((self.M, D)), 5.0 * np.ones(D)
        self.PopDec = decs(infills)
        self._set_optimum()

    def step(self):
        rng, M = self.rng, self.M
        TX, TF = decs(self.pop), objs(self.pop)
        models = []
        for i in range(M):
            dm = DaceModel(TX, TF[:, i], "regpoly0", self.theta[i], self.lob, self.upb)
            models.append(dm)
            self.theta[i] = dm.theta
        P = self.PopDec
        for _ in range(self.gmax):
            P = np.vstack([P, ga(self.problem, P, rng=rng)])
            PF = np.column_stack([m.predict(P) for m in models])
            P = P[kriging_selection(PF, self.V)]
        Fn = _norm01(TF)
        Z = Fn.min(axis=0)
        dc = np.sqrt(((Fn - Z) ** 2).sum(1))
        ddt = np.sqrt(((Fn[:, None] - Fn[None]) ** 2).sum(-1))
        ddt[ddt == 0] = np.inf
        IP = dc - ddt.min(axis=1)
        ipm = DaceModel(TX, IP, "regpoly0", self.theta0, self.lob, self.upb)
        self.theta0 = ipm.theta
        pre, mse = ipm.predict(P, mse=True)
        s = np.sqrt(mse)
        with np.errstate(all="ignore"):
            lam = (IP.min() - pre) / s
            eip = (IP.min() - pre) * norm_cdf(lam) + s * norm_pdf(lam)
        cand = P[int(np.nanargmax(eip))]
        if not np.any(np.all(TX == cand, axis=1)):
            self.pop = Population.merge(self.pop, self.evaluate(cand[None]))
        front, _ = nd_sort(objs(self.pop), None, 1)
        nd = np.where(front == 1)[0]
        ndF, ndX = _norm01(objs(self.pop)[nd]), decs(self.pop)[nd]
        ang = _angles(ndF, self.V)
        assoc = _argmin_rows(ang)
        val = ang[np.arange(len(nd)), assoc]
        sec = [ndX[np.where(assoc == i)[0][int(np.argmin(val[assoc == i]))]] for i in np.unique(assoc)]
        self.PopDec = np.unique(np.vstack([P] + ([np.array(sec)] if sec else [])), axis=0)
