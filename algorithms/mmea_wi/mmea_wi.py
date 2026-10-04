# emopylab 2026
"""MMEA-WI (weighted indicator-based evolutionary algorithm for multimodal multi-objective optimization).

Reference:
W. Li, T. Zhang, R. Wang, and H. Ishibuchi. Weighted indicator-based evolutionary algorithm for
multimodal multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(6):
1064-1078.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, pdist2, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MMEAWI': {'integer', 'multi', 'multimodal', 'real'}}


def kdis(pop, K):
    X = decs(pop)
    d = pdist2(X, X)
    np.fill_diagonal(d, np.inf)
    dn = np.sort(d, axis=0)[: int(K)].sum(axis=0)
    avg = dn.mean()
    if avg == 0:
        avg = np.inf
    return 1.0 / (1 + dn / avg)


def cal_fitness(pop, kappa):
    F = objs(pop)
    N = len(F)
    with np.errstate(all="ignore"):
        F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    I = np.max(F[:, None, :] - F[None, :, :], axis=2)
    C = np.max(np.abs(I), axis=0)
    with np.errstate(all="ignore"):
        fit = np.sum(-np.exp(-I / C[None, :] / kappa), axis=0) + 1
    return fit, I, C


def environmental_selection(pop, N, kappa, state):
    nxt = list(range(len(pop)))
    fit, I, C = cal_fitness(pop, kappa)
    X = decs(pop)
    dist = pdist2(X, X)
    sigma = np.prod(X.max(axis=0) - X.min(axis=0)) ** (1 / X.shape[1]) / X.shape[0]
    sigma *= np.exp(-state)
    with np.errstate(all="ignore"):
        w = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-dist ** 2 / (2 * sigma ** 2))
        newfit = (w * fit[None, :]).sum(axis=1) * w.sum(axis=0)
    fit = newfit
    while len(nxt) > N:
        x = int(np.argmin(fit[nxt]))
        with np.errstate(all="ignore"):
            fit = fit + np.exp(-I[nxt[x], :] / C[nxt[x]] / kappa)
        nxt.pop(x)
    pop = pop[np.array(nxt)]
    return pop, kdis(pop, N / 2)


def _double_nearest(pop, N, K):
    F, X = objs(pop), decs(pop)
    n = len(F)
    choose = np.ones(n, bool)
    if n <= K:
        return choose, np.zeros(n)
    d_obj = pdist2(F, F)
    d_dec = pdist2(X, X)
    np.fill_diagonal(d_obj, np.inf)
    np.fill_diagonal(d_dec, np.inf)

    def score():
        dn_o = np.sort(d_obj, axis=0)[:K].sum(axis=0)
        dn_d = np.sort(d_dec, axis=0)[:K].sum(axis=0)
        ao = dn_o.mean() or np.inf
        ad = dn_d.mean() or np.inf
        return 1.0 / (1 + dn_o / ao + dn_d / ad)

    fdn = score()
    while choose.sum() > N:
        dele = int(np.argmax(fdn))
        choose[dele] = False
        d_obj[dele, :] = np.inf
        d_obj[:, dele] = np.inf                      # (the decision-space matrix is deliberately left untouched)
        fdn = score()
        fdn[~choose] = -np.inf
    return choose, fdn


def update_arc(arc, off, N):
    joint = Population.merge(arc, off)
    front, _ = nd_sort(objs(joint), None, N)
    pop = joint[front == 1]
    choose, fdn = _double_nearest(pop, N, 5)
    return pop[choose], fdn[choose]


class MMEAWI(LoopAlgorithm):
    """Multimodal EA with weighted indicators: an IBEA-like fitness reweighted by a Gaussian kernel over decision
    distances (bandwidth shrinking with the run), plus an archive kept by double (objective and decision) nearest
    -neighbour density; late in the run mating is done around the archive member with the best density."""

    def __init__(self, pop_size: int = 100, kappa: float = 0.05, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.kappa = float(kappa)

    def start(self):
        self.t_gen = int(np.ceil(self.max_FE * 0.4))
        self.pop, self.pfit = environmental_selection(self.pop, self.N, self.kappa, 1)
        self.arc, self.afit = update_arc(self.pop, self.pop, self.N)
        self.P = self.pop

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        if self.FE >= self.t_gen and rng.random() > 0.5:
            p1 = int(np.argmin(self.afit))
            joint = Population.merge(self.arc, self.P)
            d = pdist2(decs(self.arc[[p1]]), decs(joint))[0]
            so = np.argsort(d, kind="stable")
            pick = so[rng.permutation(int(np.round(N / 5)))[: int(np.round(N / 10))] + 1]
            parents = Population.merge(self.arc[[p1]], joint[pick])
            off = self.evaluate(ga(pr, decs(parents), rng=rng))
        else:
            pool = tournament(2, int(np.round(N / 10)), self.pfit, rng=rng)
            off = self.evaluate(ga(pr, decs(self.P[pool]), rng=rng))
        self.P, self.pfit = environmental_selection(Population.merge(self.P, off), N, self.kappa, self.FE / self.max_FE)
        self.arc, self.afit = update_arc(self.arc, off, N)
        self.pop = self.arc
