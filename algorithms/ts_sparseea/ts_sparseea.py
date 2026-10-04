# emopylab 2026
"""TS-SparseEA (two-stage SparseEA).

Reference:
J. Jiang, F. Han, J. Wang, Q. Ling, H. Han, and Y. Wang. A two-stage evolutionary algorithm for
large-scale sparse multiobjective optimization problems. Swarm and Evolutionary Computation, 2022,
72: 101093.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, nd_sort, objs, tournament
from algorithms.community_utils.sparse_mask import nsga2_mask_selection
from core.population import Population

ALGORITHM_FLAGS = {'TSSparseEA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


def _match(rng, dec, mask, lower, upper):
    """Give every mask (in random order) the still unused decision vector whose normalised direction has the largest cosine
    with the mask; result is mapped back to the variable bounds."""
    ndec = (dec - lower) / (upper - lower)
    out = np.zeros((len(mask), dec.shape[1]))
    index = list(range(len(mask)))
    avail = list(range(len(ndec)))
    while index:
        k = int(rng.integers(0, len(index)))
        m = mask[index[k]]
        cand = ndec[avail]
        with np.errstate(all="ignore"):
            cos = (cand @ m) / (np.linalg.norm(cand, axis=1) * np.linalg.norm(m))
        h = 0 if np.all(np.isnan(cos)) else int(np.argmax(np.where(np.isnan(cos), -np.inf, cos)))
        out[index[k]] = cand[h]
        del index[k], avail[h]
    return out * (upper - lower) + lower


class TSSparseEA(LoopAlgorithm):
    """Stage one groups the variables by their single-probe rank and optimises which groups are active (binary search
    on group switches, evaluated on reference vectors) for a share ``r_eva`` of the budget; stage two runs a SparseEA-like
    search that keeps each mask matched with the most similar real-valued vector."""

    def __init__(self, pop_size: int = 100, r_eva: float = 0.1, n_group: int = 50, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.r_eva, self.n_group = float(r_eva), int(n_group)

    def _initialize_infill(self):
        rng, N, D = self.rng, self.N, self.D
        self.real = self.encoding[0] != 4
        Dec = self.lower + rng.random((N, D)) * (self.upper - self.lower) if self.real else np.ones((N, D))
        Mask = (rng.random((N, D)) < 0.5).astype(float)
        pop = self.evaluate(Dec * Mask)
        self.pop_, self.Dec, self.Mask = self._binary_group_optimization(pop, Dec, Mask)
        return self.pop_

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        _, _, _, self.front, self.crowd = nsga2_mask_selection(self.pop, self.Dec, self.Mask, self.N)
        self._set_optimum()

    def _group(self):
        rng, D, real = self.rng, self.D, self.real
        fit = np.zeros(D)
        for _ in range(1 + 4 * int(real)):
            dec = self.lower + rng.random((D, D)) * (self.upper - self.lower) if real else np.ones((D, D))
            mask = np.eye(D)
            pop = self.evaluate(dec * mask)
            C = cons(pop)
            fit += nd_sort(np.hstack([objs(pop), C]) if C.size else objs(pop), None, np.inf)[0]
        last = (pop, dec, mask)
        subs = []
        gamma = int(np.ceil(D / self.n_group))
        while np.sum(np.isfinite(fit)) > gamma:
            I = np.argsort(fit, kind="stable")[:gamma]
            subs.append(I)
            fit[I] = np.inf
        subs.append(np.where(np.isfinite(fit))[0])
        return subs, last

    def _fitfunc(self, W, subs, ref):
        rows = []
        for w in W:
            mask = np.zeros(self.D)
            for s in np.where(w == 1)[0]:
                mask[subs[s]] = 1
            rows.append(ref * mask)
        return self.evaluate(np.array(rows))

    def _mask_from_w(self, w, subs):
        m = np.zeros(self.D)
        for s in np.where(w == 1)[0]:
            m[subs[s]] = 1
        return m

    def _operator(self, pdec, pmask):
        rng = self.rng
        h = len(pmask) // 2
        P1, P2 = pmask[:h], pmask[h: 2 * h]
        n, D = P1.shape
        k = np.tile(np.arange(1, D + 1), (n, 1)) > rng.integers(1, D + 1, (n, 1))
        off = P1.copy()
        off[k] = P2[k]
        site = rng.random((n, D)) < 1.0 / D
        off[site] = 1 - off[site]
        off_dec = ga_half(self.problem, pdec, rng=rng) if self.real else np.ones((n, D))
        return off_dec, off

    def _binary_group_optimization(self, pop, Dec, Mask):
        rng, N, D, real = self.rng, self.N, self.D, self.real
        if real:
            _, _, ref, _, _ = nsga2_mask_selection(pop, Dec, Mask, self.M + 1)
        else:
            ref = np.ones((1, D))
        subs, (tpop, tdec, tmask) = self._group()
        budget = self.r_eva * self.max_FE - self.FE
        ns = len(subs)
        for i in range(len(ref)):
            W = rng.integers(0, 2, (N, ns)).astype(float)
            W[0] = np.concatenate([[1.0], np.zeros(ns - 1)])
            Pop = self._fitfunc(W, subs, ref[i])
            Pop, _, W, wfront, wcrowd = nsga2_mask_selection(Pop, np.tile(ref[i], (N, 1)), W, N)
            for _ in range(int(np.floor(budget / N / len(ref)))):
                mate = tournament(2, N, wfront, -wcrowd, rng=rng)
                _, off_w = self._operator(np.tile(ref[i], (N, 1)), W[mate])
                off_pop = self._fitfunc(off_w, subs, ref[i])
                allp = Population.merge(Pop, off_pop)
                Pop, _, W, wfront, wcrowd = nsga2_mask_selection(allp, np.tile(ref[i], (len(allp), 1)), np.vstack([W, off_w]), N)
            off_mask = np.array([self._mask_from_w(w, subs) for w in W])
            off_dec = np.array([_match(rng, Dec, off_mask[[j]], self.lower, self.upper)[0] for j in range(len(W))]) if real else np.ones((len(W), D))
            off = self.evaluate(off_dec * off_mask)
            pop, Dec, Mask, _, _ = nsga2_mask_selection(Population.merge(pop, off, tpop), np.vstack([Dec, off_dec, tdec]), np.vstack([Mask, off_mask, tmask]), N)
        return pop, Dec, Mask

    def step(self):
        N, rng = self.N, self.rng
        mate = tournament(2, N, self.front, -self.crowd, rng=rng)
        off_dec, off_mask = self._operator(self.Dec[mate], self.Mask[mate])
        if self.real:
            off_dec = _match(rng, off_dec, off_mask, self.lower, self.upper)
        off = self.evaluate(off_dec * off_mask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd = nsga2_mask_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, off_dec]), np.vstack([self.Mask, off_mask]), N)
