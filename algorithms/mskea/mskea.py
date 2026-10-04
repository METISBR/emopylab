# emopylab 2026
"""MSKEA (multi-stage knowledge-guided evolutionary algorithm).

Reference:
Z. Ding, L. Chen, D. Sun, and X. Zhang. A multi-stage knowledge-guided evolutionary algorithm for
sparse multi-objective optimization problems. Swarm and Evolutionary Computation, 2022, 73: 101119.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga_half, nd_sort, objs, pdist2, tournament, truncate_lexi
from algorithms.sparseea.sparseea import probe_and_init
from core.population import Population

ALGORITHM_FLAGS = {'MSKEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


def _truncation(F, K):
    d = pdist2(F, F)
    np.fill_diagonal(d, np.inf)
    return truncate_lexi(d, K)


def _spea2_selection(pop, Dec, Mask, N):
    _, uni = np.unique(objs(pop), axis=0, return_index=True)
    pop, Dec, Mask = pop[uni], Dec[uni], Mask[uni]
    N = min(N, len(pop))
    F = objs(pop)
    front, maxf = nd_sort(F, None, N)
    nxt = front < maxf
    with np.errstate(all="ignore"):
        f1 = F[front == 1]
        Fn = (F - f1.min(axis=0)) / (f1.max(axis=0) - f1.min(axis=0))
    last = np.where(front == maxf)[0]
    dele = _truncation(Fn[last], len(last) - N + int(nxt.sum()))
    nxt[last[~dele]] = True
    pop = pop[nxt]
    front = front[nxt]
    return pop, Dec[nxt], Mask[nxt], front, crowding(objs(pop), front)


def _ts(rng, f):
    return None if len(f) == 0 else int(tournament(2, 1, f, rng=rng)[0])


class MSKEA(LoopAlgorithm):
    """Sparse EA with multi-stage knowledge: variable scores (pv), feature votes (fv) and success votes (sv) guide
    the mask operators, moving from score-driven to vote-driven flips as the run advances."""

    def _initialize_infill(self):
        P, TPop, Dec, Mask, TDec, TMask, self.pv = probe_and_init(self)
        pop, self.Dec, self.Mask, self.front, self.crowd = _spea2_selection(
            Population.merge(P, *TPop), np.vstack([Dec] + TDec), np.vstack([Mask] + TMask), self.N)
        self.sv = np.zeros(self.D)
        self.last_num = 0
        self.fv = np.zeros(self.D)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _dec(self, ParentDec, h):
        enc = self.encoding
        if np.any(enc != 4):
            off = ga_half(self.problem, ParentDec, rng=self.rng)
            off[:, enc == 4] = 1
            return off
        return np.ones((h, self.D))

    def _operator_pvfv(self, ParentDec, ParentMask, delta):
        rng, pv, fv = self.rng, self.pv, self.fv
        n, D = ParentDec.shape
        h = n // 2
        P1, P2 = ParentMask[:h], ParentMask[h:]
        Off = P1.copy()
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where((P1[i] != 0) & (P2[i] == 0))[0]
                k = _ts(rng, -pv[idx])
                if k is not None:
                    Off[i, idx[k]] = 0
            else:
                idx = np.where((P1[i] == 0) & (P2[i] != 0))[0]
                k = _ts(rng, pv[idx])
                if k is not None:
                    Off[i, idx[k]] = P2[i, idx[k]]
        if rng.random() < (1 - delta):
            fvec = np.where(fv > 0, 1.0, fv)
            for i in range(h):
                idx = np.where(Off[i] != fvec)[0]
                if rng.random() < 0.5:
                    k = _ts(rng, -fv[idx])
                    if k is not None:
                        Off[i, idx[k]] = 1
                else:
                    k = _ts(rng, fv[idx])
                    if k is not None:
                        Off[i, idx[k]] = 0
        else:
            for i in range(h):
                if rng.random() < 0.5:
                    idx = np.where(Off[i] != 0)[0]
                    k = _ts(rng, -pv[idx])
                    if k is not None:
                        Off[i, idx[k]] = 0
                else:
                    idx = np.where(Off[i] == 0)[0]
                    k = _ts(rng, pv[idx])
                    if k is not None:
                        Off[i, idx[k]] = 1
        return self._dec(ParentDec, h), Off

    def _operator_sv(self, ParentDec, ParentMask):
        rng, sv = self.rng, self.sv
        n, D = ParentDec.shape
        h = n // 2
        P1, P2 = ParentMask[:h], ParentMask[h:]
        Off = P1.copy()
        rate0, rate1 = sv, 1 - sv
        for i in range(h):
            diff = np.where(P1[i] != P2[i])[0]
            on = Off[i, diff] != 0
            rate = np.where(on, rate1[diff], rate0[diff])
            ex = rng.random(len(diff)) < rate
            Off[i, diff[ex]] = 1 - Off[i, diff[ex]]
        mu = rng.random((h, D)) < 1.0 / D
        for i in range(h):
            if mu[i].any():
                sub = np.where(mu[i])[0]
                on = Off[i, sub] != 0
                rate = np.where(on, rate1[sub], rate0[sub])
                ex = rng.random(len(sub)) < rate
                Off[i, sub[ex]] = 1 - Off[i, sub[ex]]
        return self._dec(ParentDec, h), Off

    def step(self):
        N, rng = self.N, self.rng
        pool = tournament(2, 2 * N, self.front, -self.crowd, rng=rng)
        delta = self.FE / self.max_FE
        first = self.front == 1
        if delta < 0.618:
            fv = np.std(decs(self.pop[first]), axis=0, ddof=1) if first.sum() > 1 else np.zeros(self.D)
            b = self.encoding == 4
            fv[b] = self.Mask[first][:, b].sum(axis=0)
            self.fv = fv
        FM = self.Mask[first]
        tn = len(FM)
        vote = FM.sum(axis=0)
        self.sv = (self.last_num / (self.last_num + tn)) * self.sv + (tn / (self.last_num + tn)) * (vote / tn)
        self.last_num = tn
        if delta < 0.618:
            self.pv = self.pv * (1 - self.sv) * np.sqrt(delta) + self.pv
        pd, pm = self.Dec[pool], self.Mask[pool]
        if delta / 0.618 < 0.618:
            OffDec, OffMask = self._operator_pvfv(pd, pm, delta)
        elif delta < 0.618:
            if rng.random() < 0.5:
                OffDec, OffMask = self._operator_sv(pd, pm)
            else:
                OffDec, OffMask = self._operator_pvfv(pd, pm, delta)
        else:
            OffDec, OffMask = self._operator_sv(pd, pm)
        off = self.evaluate(OffDec * OffMask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd = _spea2_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), N)
