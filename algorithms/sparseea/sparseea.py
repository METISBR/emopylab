# emopylab 2026
"""SparseEA (evolutionary algorithm for sparse multi-objective optimization problems).

Reference:
Y. Tian, X. Zhang, C. Wang, and Y. Jin. An evolutionary algorithm for large-scale sparse multi-
objective optimization problems. IEEE Transactions on Evolutionary Computation, 2020, 24(2):
380-393.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, ga_half, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'SparseEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


def _env_selection(pop, Dec, Mask, N):
    _, uni = np.unique(objs(pop), axis=0, return_index=True)
    pop, Dec, Mask = pop[uni], Dec[uni], Mask[uni]
    N = min(N, len(pop))
    C = cons(pop)
    front, maxf = nd_sort(objs(pop), C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(objs(pop), front)
    last = np.where(front == maxf)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], Dec[nxt], Mask[nxt], front[nxt], cd[nxt]


def probe_and_init(algo):
    """Variable-score probing (one solution per single active variable) and the sparse random initial population.

    Returns ``(pop, probe_pops, Dec, Mask, probe_decs, probe_masks, variable_scores)``."""
    N, D, rng = algo.N, algo.D, algo.rng
    lo, up, enc = algo.lower, algo.upper, algo.encoding
    b = enc == 4
    TDec, TMask, TPop = [], [], []
    Fit = np.zeros(D)
    for _ in range(1 + 4 * int(np.any(~b))):
        Dec = lo + rng.random((D, D)) * (up - lo)
        Dec[:, b] = 1
        Mask = np.eye(D)
        P = algo.evaluate(Dec * Mask)
        TDec.append(Dec), TMask.append(Mask), TPop.append(P)
        C = cons(P)
        Fit = Fit + nd_sort(np.hstack([objs(P), C]) if C.size else objs(P), None, np.inf)[0]
    Dec = lo + rng.random((N, D)) * (up - lo)
    Dec[:, b] = 1
    Mask = np.zeros((N, D))
    for i in range(N):
        Mask[i, tournament(2, int(np.ceil(rng.random() * D)), Fit, rng=rng)] = 1
    return algo.evaluate(Dec * Mask), TPop, Dec, Mask, TDec, TMask, Fit


class SparseEA(LoopAlgorithm):
    """Evolutionary algorithm for sparse problems: a real-valued decision vector plus a binary mask, whose
    bits are flipped with tournament pressure on the variable scores gathered at initialization."""

    def _initialize_infill(self):
        P, TPop, Dec, Mask, TDec, TMask, self.fitness = probe_and_init(self)
        pop, self.Dec, self.Mask, self.front, self.crowd = _env_selection(
            Population.merge(P, *TPop), np.vstack([Dec] + TDec), np.vstack([Mask] + TMask), self.N)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _operator(self, ParentDec, ParentMask):
        rng, Fit = self.rng, self.fitness
        n, D = ParentDec.shape
        h = n // 2
        P1, P2 = ParentMask[:h], ParentMask[h:]
        Off = P1.copy()

        def ts(f):
            return None if len(f) == 0 else int(tournament(2, 1, f, rng=rng)[0])

        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where((P1[i] != 0) & (P2[i] == 0))[0]
                k = ts(-Fit[idx])
                if k is not None:
                    Off[i, idx[k]] = 0
            else:
                idx = np.where((P1[i] == 0) & (P2[i] != 0))[0]
                k = ts(Fit[idx])
                if k is not None:
                    Off[i, idx[k]] = P2[i, idx[k]]
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where(Off[i] != 0)[0]
                k = ts(-Fit[idx])
                if k is not None:
                    Off[i, idx[k]] = 0
            else:
                idx = np.where(Off[i] == 0)[0]
                k = ts(Fit[idx])
                if k is not None:
                    Off[i, idx[k]] = 1
        enc = self.encoding
        if np.any(enc != 4):
            OffDec = ga_half(self.problem, ParentDec, rng=rng)
            OffDec[:, enc == 4] = 1
        else:
            OffDec = np.ones((h, D))
        return OffDec, Off

    def step(self):
        pool = tournament(2, 2 * self.N, self.front, -self.crowd, rng=self.rng)
        OffDec, OffMask = self._operator(self.Dec[pool], self.Mask[pool])
        off = self.evaluate(OffDec * OffMask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd = _env_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), self.N)
