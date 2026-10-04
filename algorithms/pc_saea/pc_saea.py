# emopylab 2026
"""PC-SAEA (pairwise comparison based surrogate-assisted evolutionary algorithm).

Reference:
Y. Tian, J. Hu, C. He, H. Ma, L. Zhang, and X. Zhang. A pairwise comparison based surrogate-assisted
evolutionary algorithm for expensive multi-objective optimization. Swarm and Evolutionary
Computation, 2023, 80: 101323.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import spea
from algorithms.community_utils.base import LoopAlgorithm, decs, ga, objs
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population

ALGORITHM_FLAGS = {'PCSAEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


def _dominance(F):
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    gt = (F[:, None, :] > F[None, :, :]).any(-1)
    return lt & ~gt


def pc_fitness(F, X, rate):
    """Training data for the pairwise model: best quarter (class 2) and worst quarter (class 1) by a blend of strength
    rank and density, the fitness appended as last column; also the best rows and the rows around the class border."""
    N = len(F)
    with np.errstate(all="ignore"):
        F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    k = int(np.floor(np.sqrt(N)))
    sde = np.zeros(N)
    for i in range(N):
        S = np.maximum(F, F[i])
        d = np.sort(np.sqrt(((F[i] - S) ** 2).sum(1)))
        sde[i] = 2.0 / (d[k] + 2)
    dom = _dominance(F)
    S = dom.sum(1)
    R = np.array([S[dom[:, i]].sum() for i in range(N)], float)
    o = sde[:, None]
    with np.errstate(all="ignore"):
        cosd = 1 - (o @ o.T) / (np.abs(o) @ np.abs(o).T)
    np.fill_diagonal(cosd, np.inf)
    Dn = 1.0 / (np.sort(cosd, axis=1)[:, k - 1] + 2)
    with np.errstate(all="ignore"):
        R = (R - R.min()) / (R.max() - R.min())
    fit = rate * R + (1 - rate) * Dn
    idx = np.argsort(fit, kind="stable")
    q, h = int(np.ceil(N / 4)), int(np.ceil(N / 2))
    sel = np.concatenate([idx[:q], idx[len(idx) - (h - q):]])
    Inp = np.hstack([X[sel], fit[sel][:, None]])
    out = np.ones(h)
    out[:q] = 2
    e = int(np.floor(N / 8))
    return Inp, out, Inp[:q], Inp[q - 1 - e:q + e]


class PairwisePNN:
    """Probabilistic neural network on concatenated pairs [x_i, x_j]: class 2 when x_i is better than x_j, else 1."""

    def __init__(self, spread):
        self.b = 0.8326 / spread

    def train(self, X, D):
        N = len(X)
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
        self.W = np.hstack([X[ii, :D], X[jj, :D]])
        self.T = np.where(X[ii, -1] < X[jj, -1], 2, 1)
        self.w2 = self.W @ np.ones(self.W.shape[1])
        self.ww = (self.W ** 2).sum(1)
        return self

    def _cls(self, Q):
        d = np.sqrt(np.maximum((Q ** 2).sum(1)[:, None] + self.ww[None] - 2 * Q @ self.W.T, 0))
        a = np.exp(-(d * self.b) ** 2)
        s1, s2 = a[:, self.T == 1].sum(1), a[:, self.T == 2].sum(1)
        return np.where(s2 > s1, 2, 1)

    def lastpredict(self, X, D, pref, flag):
        n, nP = len(X), len(pref)
        P = pref[(np.arange(n) + 1) % nP, :D]
        dec = X[:, :D]
        Y = self._cls(np.hstack([dec, P]))
        y = self._cls(np.hstack([P, dec]))
        res = Y - y
        if flag == 0:
            return res.astype(float)
        Y = Y.astype(float)
        Y[res == 0] = 1.5
        return Y


class PCSAEA(LoopAlgorithm):
    """A pairwise-comparison PNN predicts which of two solutions is better; GA offspring compared against the current best
    quarter are refined for ``gmax`` surrogate predictions (ascending or descending, depending on the model's accuracy on
    held-out pairs) and the best-labelled ones are evaluated; the population is kept by SPEA2-style selection."""

    def __init__(self, pop_size: int = 100, delta: float = 0.8, gmax: int = 3000, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta, self.gmax = float(delta), int(gmax)

    def _initialize_infill(self):
        n = max(11 * self.D - 1, self.N)
        return self.evaluate(self.lower + lhs_design(self.rng, n, self.D) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.P = infills
        self.pop = infills
        self._set_optimum()

    def _ga(self, X):
        return ga(self.problem, X, (1, 15, 1, 5), rng=self.rng)

    def step(self):
        rng, D = self.rng, self.D
        Inp, out, Pa, Pmid = pc_fitness(objs(self.P), decs(self.P), self.FE / self.max_FE)
        i1, i0 = np.where(out > 1)[0], np.where(out <= 1)[0]
        K = np.concatenate([rng.permutation(i1)[: int(np.ceil(0.75 * len(i1)))], rng.permutation(i0)[: int(np.ceil(0.75 * len(i0)))]])
        te = np.setdiff1d(np.arange(len(out)), K)
        net = PairwisePNN(0.1925).train(Inp[K], D)
        pre = net.lastpredict(Inp[te], D, Pmid, 1)
        valid = pre != 1.5
        e1 = (out[te][valid] == pre[valid]).sum() / len(te)
        e2 = (out[te][valid] != pre[valid]).sum() / len(te)
        Pa = Pa[:, :D]
        nxt = self._ga(np.vstack([decs(self.P), Pa]))
        lab = net.lastpredict(nxt, D, Pa, 0)
        ln = len(Pa)
        wmax = self.gmax // ln
        if e1 < 1 - self.delta or e2 < 1 - self.delta:
            desc = e1 < 1 - self.delta
            gN, gL = np.zeros((wmax, D)), np.zeros(wmax)
            for i in range(wmax):
                idx = np.argsort(-lab if desc else lab, kind="stable")
                gN[i], gL[i] = nxt[idx[0]], lab[idx[0]]
                par = np.vstack([nxt[idx[:ln]], Pa])
                nxt = self._ga(par[rng.permutation(len(par))])
                lab = net.lastpredict(nxt, D, Pa, 0)
            good = gL >= 0.95 if desc else gL <= -0.95
            if good.sum() == 0 or good.sum() > ln // 2:
                nxt = gN[np.argsort(-gL if desc else gL, kind="stable")[: ln // 2]]
            else:
                nxt = gN[good]
        else:
            nxt = nxt[[int(rng.integers(0, len(nxt)))]]
        if len(nxt):
            self.pop = Population.merge(self.pop, self.evaluate(nxt))
        self.P, _ = spea.select(self.pop, self.N, use_cons=False)
