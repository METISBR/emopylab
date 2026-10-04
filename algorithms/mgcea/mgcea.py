# emopylab 2026
"""MGCEA (multi-granularity clustering based evolutionary algorithm).

Reference:
Y. Tian, S. Shao, G. Xie, and Y. Jin. A multi-granularity clustering based evolutionary algorithm
for large-scale sparse multi-objective optimization. Swarm and Evolutionary Computation, 2024, 84:
101453.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, ga_half, kmeans, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MGCEA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


def _spea2_selection(pop, Dec, Mask, N):
    _, uni = np.unique(objs(pop), axis=0, return_index=True)
    pop, Dec, Mask = pop[uni], Dec[uni], Mask[uni]
    N = min(N, len(pop))
    fit = cal_fitness(objs(pop))
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt], Dec[nxt], Mask[nxt], fit[nxt]


def _cluster_label(fit_init, labels):
    n1, n2 = int(np.sum(labels == 1)), int(np.sum(labels == 2))
    v1, v2 = fit_init[labels == 1].sum(), fit_init[labels == 2].sum()
    out = labels.copy()
    if v1 < v2:
        rate = n1 / (n1 + n2)
        out[labels == 1], out[labels == 2] = 11, 12
    else:
        rate = n2 / (n1 + n2)
        out[labels == 1], out[labels == 2] = 12, 11
    return rate, out


def update_layer(sparse_rate, stage, fitness, algo, Mask):
    D, rng = algo.D, algo.rng
    group = np.ceil(11 - stage) / 100 * D
    group = np.ceil(sparse_rate * 10 * 1 * group)
    if Mask is None or Mask.sum() == 0:
        idx = np.argsort(fitness + rng.random(D), kind="stable")
    else:
        idx = np.argsort(fitness + np.sum(Mask == 0, axis=0) / 100000, kind="stable")
    with np.errstate(all="ignore"):
        layer_of_rank = np.ceil(np.arange(1, D + 1) / group)
    layer = np.zeros(D)
    layer[idx] = layer_of_rank
    return layer, layer.max()


class MGCEA(LoopAlgorithm):
    """Multi-granularity sparse EA: variable scores are clustered into 'sparse' and 'dense' groups, ranked into
    layers whose granularity coarsens over ten stages, and the mask operator flips whole layers of variables."""

    def _initialize_infill(self):
        N, D, rng = self.N, self.D, self.rng
        lo, up, enc = self.lower, self.upper, self.encoding
        real = bool(np.any(enc == 1))
        TDec, TMask, TPop = [], [], []
        if real:
            Fit = np.zeros((5, D))
            interval = (up - lo) / 5
            for i in range(5):
                for _ in range(2):
                    Dec = (lo + interval * i) + rng.random((D, D)) * interval
                    Mask = np.eye(D)
                    P = self.evaluate(Dec * Mask)
                    TDec.append(Dec), TMask.append(Mask), TPop.append(P)
                    C = cons(P)
                    Fit[i] += nd_sort(np.hstack([objs(P), C]) if C.size else objs(P), None, np.inf)[0]
            if D > 2000:
                allp = Population.merge(*TPop)
                pick = rng.permutation(len(allp))[:D]
                Dm, Mm = np.vstack(TDec)[pick], np.vstack(TMask)[pick]
                TPop, TDec, TMask = [allp[pick]], [Dm], [Mm]
            init = Fit.sum(axis=0)
        else:
            Dec = np.ones((D, D))
            Mask = np.eye(D)
            P = self.evaluate(Dec * Mask)
            TDec.append(Dec), TMask.append(Mask), TPop.append(P)
            C = cons(P)
            init = nd_sort(np.hstack([objs(P), C]) if C.size else objs(P), None, np.inf)[0]
        labels = kmeans(init.reshape(-1, 1), 2, rng) + 1
        self.sparse_rate, self.fitness = _cluster_label(init, labels)
        if not real:
            Mask = np.zeros((N, D))
            Dec = np.ones((N, D))
            for i in range(N):
                Mask[i, tournament(2, int(np.ceil(rng.random() * D)), init, rng=rng)] = 1
            P = self.evaluate(Dec * Mask)
            TDec.append(Dec), TMask.append(Mask), TPop.append(P)
        pop, self.Dec, self.Mask, self.fit_spea2 = _spea2_selection(
            Population.merge(*TPop) if len(TPop) > 1 else TPop[0], np.vstack(TDec), np.vstack(TMask), N)
        self.near_stage = int(np.ceil(self.FE / (self.max_FE / 10)))
        self.layer, self.layer_max = update_layer(self.sparse_rate, self.near_stage, self.fitness, self, None)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _operator(self, ParentDec, ParentMask):
        rng, D = self.rng, self.D
        n = len(ParentDec)
        h = n // 2
        P1, P2 = ParentMask[:h], ParentMask[h:]
        Off = P1.copy()
        s1, s2 = P1.sum(axis=1), P2.sum(axis=1)
        with np.errstate(all="ignore"):
            rate = s1 / (s1 + s2)
        idx = rng.random((h, D)) > (rate / 2)[:, None]
        Off[idx] = P2[idx]
        for i in range(h):
            up, down = 1, self.layer_max
            j = 1
            while j <= self.layer_max:
                lay_up = np.where(self.layer == up)[0]
                tgt_up = lay_up[Off[i, lay_up] == 0]
                lay_dn = np.where(self.layer == down)[0]
                tgt_dn = lay_dn[Off[i, lay_dn] == 1]
                if rng.random() < 0.5:
                    if len(tgt_up) and rng.random() < 0.5:
                        pick = rng.permutation(len(tgt_up))[: int(np.ceil(len(tgt_up) / 2))]
                        Off[i, tgt_up[pick]] = 1
                    if rng.random() < 0.5:
                        up += 1
                    else:
                        break
                else:
                    if len(tgt_dn) and rng.random() < 0.5:
                        pick = rng.permutation(len(tgt_dn))[: int(np.ceil(len(tgt_dn) / 2))]
                        Off[i, tgt_dn[pick]] = 0
                    if rng.random() < 0.5:
                        down -= 1
                    else:
                        break
                if up >= down:
                    break
                j += 1
        enc = self.encoding
        if np.any(enc != 4):
            OffDec = ga_half(self.problem, ParentDec, rng=rng)
            OffDec[:, enc == 4] = 1
        else:
            OffDec = np.ones((h, D))
        return OffDec, Off

    def step(self):
        N, rng = self.N, self.rng
        pool = tournament(2, 2 * N, self.fit_spea2, rng=rng)
        stage = int(np.ceil(self.FE / (self.max_FE / 10)))
        if stage != self.near_stage:
            self.near_stage = stage
            self.layer, self.layer_max = update_layer(self.sparse_rate, stage, self.fitness, self, self.Mask)
        OffDec, OffMask = self._operator(self.Dec[pool], self.Mask[pool])
        off = self.evaluate(OffDec * OffMask)
        self.pop, self.Dec, self.Mask, self.fit_spea2 = _spea2_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), N)
