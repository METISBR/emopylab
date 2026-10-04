# emopylab 2026
"""PIMD (probability and mapping crowding distance).

Reference:
Y. Li, W. Li, Y. Zhao, and S. Li. An infill sampling criterion based on improvement of probability
and mapping crowding distance for expensive multi/many-objective optimization. Engineering
Applications of Artificial Intelligence, 2024, 133: 108616.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import nsga3_ref
from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, uniform_point
from algorithms.community_utils.dace import DaceModel, norm_cdf
from core.population import Population

ALGORITHM_FLAGS = {'PIMD': {'expensive', 'integer', 'many', 'multi', 'real'}}

_CP = np.array([0.27, 0.36, 0.43, 0.5, 0.57, 0.66, 0.75, 0.86, 1, 1.15, 1.35, 1.6, 2, 2.4, 3.1, 4.2, 6.5])


def shape_estimate(F, N):
    """Lp exponent whose Lp-norm of the normalised first front is the most uniform (box-plot denoised)."""
    front, _ = nd_sort(F, None, N)
    F = F[front <= 1]
    if len(F) < 20:
        return 1.0
    n = len(F)
    with np.errstate(all="ignore"):
        F = (F - F.min(0)) / (F.max(0) - F.min(0))
    vp = np.zeros(len(_CP))
    for i, p in enumerate(_CP):
        with np.errstate(all="ignore"):
            g = (F ** p).sum(1) ** (1 / p)
        t = np.sort(g)
        q1, q3 = t[max(int(n * 0.25), 1) - 1], t[max(int(n * 0.75), 1) - 1]
        g = g[~(g > q3 + 1.5 * (q3 - q1))]
        vp[i] = np.std(g / g.max(), ddof=1) if len(g) > 1 else 0.0
    return float(_CP[int(np.nanargmin(vp))])


def _nsga3_select(F, N, W, zmin, rng):
    front, maxf = nd_sort(F, None, N)
    nxt = front < maxf
    last = np.where(front == maxf)[0]
    ch = nsga3_ref.last_selection(F[nxt], F[last], N - int(nxt.sum()), W, zmin, rng)
    nxt[last[ch]] = True
    return nxt


class PIMD(LoopAlgorithm):
    """NSGA-III evolved on Kriging surrogates; among the uncertain survivors, the first Pareto front of two infill indicators
    (probability of dominating an evaluated non-dominated solution, and distance to them after mapping onto the estimated
    Lp-shaped front) is evaluated, at most ``eta`` of them (always including the most promising one)."""

    def __init__(self, pop_size: int = 100, wmax: int = 15, eta: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax, self.eta = int(wmax), int(eta)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.NI = self.N
        P, _ = UniformPoint(self.NI, self.D, "Latin", rng=self.rng)
        self.W, _ = uniform_point(self.N, self.M)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop, self.P = infills, infills
        self.zmin = objs(infills).min(0)
        self.theta = 5.0 * np.ones((self.M, self.D))
        self._set_optimum()

    def step(self):
        rng, NI, D = self.rng, self.NI, self.D
        AF = objs(self.pop)
        fr, _ = nd_sort(AF, None, NI)
        comp = AF[fr <= 1]
        Dec, Obj = decs(self.P), objs(self.P)
        Mse = np.zeros_like(Obj)
        lp = shape_estimate(Obj, self.N)
        TX, TY = decs(self.pop), AF
        dist = np.unique(np.round(TX * 1e6) / 1e6, axis=0, return_index=True)[1]
        TX, TY = TX[dist], TY[dist]
        models = []
        for i in range(self.M):
            m = DaceModel(TX, TY[:, i], "regpoly0", self.theta[i], 1e-5 * np.ones(D), 100 * np.ones(D))
            self.theta[i] = m.theta
            models.append(m)
        for _ in range(self.wmax):
            n = max(Dec.shape)                       # the reference draws length(Dec) parents
            od = ga(self.problem, Dec[rng.integers(0, len(Dec), n)], rng=rng)
            pr = [m.predict(od, mse=True) for m in models]
            of, om = np.column_stack([p[0] for p in pr]), np.column_stack([p[1] for p in pr])
            aO, aD, aM = np.vstack([Obj, of]), np.vstack([Dec, od]), np.vstack([Mse, om])
            self.zmin = np.minimum(self.zmin, of.min(0))
            nxt = _nsga3_select(aO, NI, self.W, self.zmin, rng)
            Dec, Obj, Mse = aD[nxt], aO[nxt], aM[nxt]
        keep = Mse.sum(1) >= 1e-15
        Dec, Obj, Mse = Dec[keep], Obj[keep], Mse[keep]
        if len(Dec):
            with np.errstate(all="ignore"):
                pr_dom = np.array([-np.max(np.prod(norm_cdf((comp - Obj[i]) / np.sqrt(np.maximum(Mse[i], 0))), 1)) for i in range(len(Obj))])
            allo = np.vstack([Obj, comp])
            z, zn = allo.min(0), allo.max(0)
            with np.errstate(all="ignore"):
                nO, nC = (Obj - z) / (zn - z), (comp - z) / (zn - z)
                tO = (nO + 1e-6) / ((np.abs(nO + 1e-6) ** lp).sum(1) ** (1 / lp))[:, None]
                tC = (nC + 1e-6) / ((np.abs(nC + 1e-6) ** lp).sum(1) ** (1 / lp))[:, None]
            di = -np.sqrt(((tC[:, None] - tO[None]) ** 2).sum(-1)).min(0)
            f1, _ = nd_sort(np.column_stack([di, pr_dom]), None, 1)
            first = f1 == 1
            new, sel = np.unique(Dec[first], axis=0, return_index=True)
            if len(new) > self.eta:
                ind = int(np.argmin(pr_dom[first][sel]))
                tmp = rng.permutation(len(new))[: self.eta]
                if ind not in tmp:
                    tmp[0] = ind
                new = new[tmp]
            self.pop = Population.merge(self.pop, self.evaluate(new))
        AF = objs(self.pop)
        self.zmin = AF.min(0)
        self.P = self.pop[_nsga3_select(AF, NI, self.W, self.zmin, rng)]
