# emopylab 2026
"""CSEA (classification based surrogate-assisted evolutionary algorithm).

Reference:
L. Pan, C. He, Y. Tian, H. Wang, X. Zhang, and Y. Jin. A classification based surrogate-assisted
evolutionary algorithm for expensive many-objective optimization. IEEE Transactions on Evolutionary
Computation, 2019, 23(1): 74-88.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs
from algorithms.community_utils.nn import BNClassifierNet
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population

ALGORITHM_FLAGS = {'CSEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


def _radar_grid(P, div):
    N, M = P.shape
    th = np.arange(M) * 2 * np.pi / M
    with np.errstate(all="ignore"):
        R = np.column_stack([(P * np.cos(th)).sum(1) / P.sum(1), (P * np.sin(th)).sum(1) / P.sum(1)])
        R = (R + 1) / 2
        NR = (R - R.min(axis=0)) / (R.max(axis=0) - R.min(axis=0))
    G = np.floor(NR * div)
    G[G >= div] = div - 1
    _, site = np.unique(G, axis=0, return_inverse=True)
    return site.ravel(), R


def _last_selection(F, choose, div, k):
    n = len(F)
    with np.errstate(all="ignore"):
        cos = 1 - F.sum(1) / (np.linalg.norm(F, axis=1) * np.sqrt(F.shape[1]))
        pbi = np.linalg.norm(F, axis=1) * np.sqrt(1 - (1 - cos) ** 2)
    choose = choose.copy()
    choose[int(np.nanargmin(pbi))] = True
    con = F.sum(1)
    con = con / con.max()
    site, R = _radar_grid(F, div)
    rd = np.sqrt(((R[:, None] - R[None]) ** 2).sum(-1))
    np.fill_diagonal(rd, np.inf)
    crowd = np.bincount(site[choose], minlength=site.max() + 1).astype(float)
    while choose.sum() < k:
        rem = np.where(~choose)[0]
        rg = np.unique(site[rem])
        best = rg[crowd[rg] == crowd[rg].min()]
        cur = rem[np.isin(site[rem], best)]
        near = rd[np.ix_(cur, np.where(choose)[0])].min(axis=1) if choose.any() else np.full(len(cur), np.inf)
        b = cur[int(np.argmin(0.1 * F.shape[1] * con[cur] - near))]
        choose[b] = True
        crowd[site[b]] += 1
    return choose


def ref_select(pop, k):
    k = min(k, len(pop))
    F = objs(pop)
    front, maxf = nd_sort(F, None, k)
    nxt = np.where(front <= maxf)[0]
    pmin, pmax = F.min(axis=0) + 1e-6, F.max(axis=0)
    if np.all(pmax > pmin):
        F = (F - pmin) / (pmax - pmin)
    ch = _last_selection(F[nxt], front[nxt] < maxf, int(np.ceil(np.sqrt(k))), k)
    return pop[nxt[ch]]


def get_output(F, R):
    out = np.ones(len(F), bool)
    for r in R:
        out &= np.any(F <= r, axis=1)
    return out


class CSEA(LoopAlgorithm):
    """A neural classifier learns whether a solution is dominated by none of ``k`` reference solutions; depending on its
    measured error on held-out data, GA offspring are screened for predicted-good (or predicted-bad) solutions over up to
    ``gmax`` surrogate predictions, and the screened ones are evaluated for real."""

    def __init__(self, pop_size: int = 100, k: int = 6, gmax: int = 3000, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.k, self.gmax = int(k), int(gmax)

    def _initialize_infill(self):
        n = min(11 * self.D - 1, 109)
        return self.evaluate(self.lower + lhs_design(self.rng, n, self.D) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.P = infills
        self.pop = infills          # Arc: all evaluated solutions
        self._set_optimum()

    def _offspring(self, X):
        return ga(self.problem, X, (1, 15, 1, 5), rng=self.rng)

    def step(self):
        rng = self.rng
        ref = ref_select(self.P, self.k)
        X, y = decs(self.P), get_output(objs(self.P), objs(ref)).astype(float)
        rr = y.mean()
        tr = min(rr, 1 - rr) * 0.5
        i1, i0 = np.where(y > 0.5)[0], np.where(y <= 0.5)[0]
        K = np.concatenate([rng.permutation(i1)[: int(np.ceil(0.75 * len(i1)))], rng.permutation(i0)[: int(np.ceil(0.75 * len(i0)))]])
        test = np.setdiff1d(np.arange(len(X)), K)
        net = BNClassifierNet(self.D, int(np.ceil(self.D * 2)), rng).fit(X[K], y[K])
        pre, to = net.predict(X[test]), y[test]
        good = to == 1
        with np.errstate(all="ignore"):
            p0 = np.abs(to[good] - pre[good]).sum() / good.sum()
            p1 = np.abs(to[~good] - pre[~good]).sum() / (~good).sum()
        rdec = decs(ref)
        nxt = self._offspring(np.vstack([X, rdec]))
        lab = net.predict(nxt)
        a, b = tr, 1 - tr
        if p0 < 0.4 or (p1 < a and p0 < b):
            i = 0
            while i < self.gmax:
                inp = nxt[np.argsort(-lab, kind="stable")[: len(ref)]]
                nxt = self._offspring(np.vstack([inp, rdec]))
                lab = net.predict(nxt)
                i += len(nxt)
            nxt = nxt[lab > 0.9]
        elif p0 > b and p1 < a:
            nxt = nxt[[int(rng.permutation(len(nxt))[0])]]
        elif p1 > b:
            i = 0
            while i < self.gmax:
                inp = nxt[np.argsort(lab, kind="stable")[: len(ref)]]
                nxt = self._offspring(np.vstack([inp, rdec]))
                lab = net.predict(nxt)
                i += len(nxt)
            nxt = nxt[lab < 0.1]
        else:
            nxt = nxt[[int(rng.integers(0, len(nxt)))]]
        if len(nxt):
            self.pop = Population.merge(self.pop, self.evaluate(nxt))
        self.P = ref_select(self.pop, self.N)
