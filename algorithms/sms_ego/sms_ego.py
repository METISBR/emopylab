# emopylab 2026
"""SMS-EGO (s-metric-selection-based efficient global optimization).

Reference:
W. Ponweiser, T. Wagner, D. Biermann, and M. Vincze. Multiobjective optimization on a limited budget
of evaluations using model-assisted S-metric selection. Proceedings of the International Conference
on Parallel Problem Solving from Nature, 2008, 784-794.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs
from algorithms.community_utils.dace import DaceModel, norm_cdf
from core.population import Population

ALGORITHM_FLAGS = {'SMSEGO': {'expensive', 'integer', 'multi', 'real'}}


def _cal_hv(rng, P, ref):
    """Monte-Carlo hypervolume (10000 samples) of the points that do not exceed the reference point."""
    P = P[~np.any(P > ref, axis=1)]
    if len(P) == 0:
        return 0.0
    lo = P.min(axis=0)
    S = rng.uniform(lo, ref, (10000, P.shape[1]))
    dom = np.zeros(10000, bool)
    for p in P:
        dom |= np.all(p <= S, axis=1)
    return float(np.prod(ref - lo) * dom.sum() / 10000)


def _contribution(rng, nleft, P, ypot):
    N, M = P.shape
    c = 1 - 1 / 2 ** M
    eps = (P.max(axis=0) - P.min(axis=0)) / len(P) + c * nleft / 500
    Pe = P - eps
    ref = P.max(axis=0) * 1.1
    hv_sum = _cal_hv(rng, P, ref) if M > 2 else 0.0
    fit = np.zeros(len(ypot))
    for i, y in enumerate(ypot):
        strong = np.where(~np.any(y <= P, axis=1) & np.any(y >= P, axis=1))[0]
        if len(strong):
            idx, dom = 2, strong
        else:
            stronge = np.where(~np.any(y <= Pe, axis=1) & np.any(y >= Pe, axis=1))[0]
            idx, dom = (3, stronge) if len(stronge) else (1, None)
        if idx == 1:
            if M == 2:
                new = np.vstack([y, P])
                rank = np.lexsort((new[:, 1], new[:, 0]))
                j = int(np.where(rank == 0)[0][0])
                if j == 0 or j == len(rank) - 1:
                    fit[i] = -np.prod(y - ref) if (np.any(y <= ref) and not np.any(y >= ref)) else 0.0
                else:
                    fit[i] = -(new[rank[j + 1], 0] - new[rank[j], 0]) * (new[rank[j - 1], 1] - new[rank[j], 1])
            else:
                fit[i] = hv_sum - _cal_hv(rng, np.vstack([P, y]), ref)
        elif idx == 2:
            fit[i] = sum(-1 + np.prod(1 + (y - P[k])) for k in dom)
        else:
            flag = (y - P[dom]) > 0
            fit[i] = sum(-1 + np.prod(1 + (y - P[k]) * flag[n]) for n, k in enumerate(dom))
    return fit


class SMSEGO(LoopAlgorithm):
    """Each objective gets a Kriging model; a GA searches the lower confidence bounds of the models and the candidate
    with the best (penalised) hypervolume contribution with respect to the current non-dominated set is evaluated."""

    def __init__(self, pop_size: int = 100, wmax: int = 10000, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax = int(wmax)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        n = 11 * self.D - 1
        P, _ = UniformPoint(n, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.theta = 5.0 * np.ones((self.M, self.D))
        self.alpha = 1.0 / float(norm_cdf(0.5 + 1 / 2 ** self.M))
        self._set_optimum()

    def step(self):
        rng, D, M = self.rng, self.D, self.M
        pop = self.pop
        index = np.unique(decs(pop), axis=0, return_index=True)[1]
        pop = pop[index]
        pdec, pobj = decs(pop), objs(pop)
        models = []
        for i in range(M):
            dm = DaceModel(pdec, pobj[:, i], "regpoly1", self.theta[i], 1e-5 * np.ones(D), 20 * np.ones(D))
            models.append(dm)
            self.theta[i] = dm.theta
        nd = pobj[nd_sort(pobj, None, 1)[0] == 1]
        w = 0
        cand = pdec
        while w < self.wmax:
            cand = np.vstack([cand, ga(self.problem, cand, rng=rng)])
            preds = [m.predict(cand, mse=True) for m in models]
            F = np.column_stack([p[0] for p in preds])
            s2 = np.column_stack([p[1] for p in preds])
            ypot = F - self.alpha * np.sqrt(np.maximum(s2, 0))
            fit = _contribution(rng, self.max_FE - self.FE, nd, ypot)
            order = np.argsort(fit, kind="stable")[: len(cand) // 2]
            cand, fit = cand[order], fit[order]
            w += len(cand)
        best = int(np.argmin(fit))
        self.pop = Population.merge(pop, self.evaluate(cand[[best]]))
