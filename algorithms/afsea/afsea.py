# emopylab 2026
"""AFSEA (adjoint feature-selection-based evolutionary algorithm).

Reference:
P. Zhang, H. Yin, Y. Tian, and X. Zhang. An adjoint feature-selection- based evolutionary algorithm
for sparse large-scale multiobjective optimization. Complex & Intelligent Systems, 2025, 11: 127.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, ga_half, nd_sort, objs, tournament
from algorithms.community_utils.spea import truncation
from core.population import Population

ALGORITHM_FLAGS = {'AFSEA': {'binary', 'integer', 'multi', 'real', 'sparse'}}


def _first_unique(F):
    """Indices of the first occurrence of every distinct objective row, in lexicographic order of the rows."""
    return np.unique(F, axis=0, return_index=True)[1]


def _relief(algo, pop, front):
    """Feature relevance from the nearest same-class / other-class solution (class = first front or not)."""
    N, D = len(pop), algo.D
    labels = (front == 1).astype(int)
    if np.all(labels == 1):
        return np.zeros(D)
    X = decs(pop)
    dist = np.sqrt(np.maximum(np.sum(X * X, 1)[:, None] + np.sum(X * X, 1)[None, :] - 2.0 * X @ X.T, 0.0))
    np.fill_diagonal(dist, np.inf)
    data = (X - algo.lower) / (algo.upper - algo.lower)
    delta = np.zeros(D)
    for i in range(N):
        same = np.where(labels == labels[i])[0]
        other = np.where(labels != labels[i])[0]
        hit = same[int(np.argmin(dist[same, i]))]
        miss = other[int(np.argmin(dist[other, i]))]
        delta += np.abs(data[miss] - data[i]) ** 2 - np.abs(data[hit] - data[i]) ** 2
    return delta


class AFSEA(LoopAlgorithm):
    """Feature-selection view of sparse optimisation: single-variable probes give every variable a non-domination score
    that seeds the sparse masks, and a Relief-style relevance measure adapts which variables the mask operators favour."""

    def _initialize_infill(self):
        rng, D, N = self.rng, self.D, self.N
        enc = self.encoding
        real = bool(np.any(enc != 4))
        lo, up = self.lower, self.upper
        TDec, TMask, TPop = [], [], []
        Fit = np.zeros(D + 1)
        for _ in range(1 + 4 * int(real)):
            Dec = lo + rng.random((D, D)) * (up - lo)
            Dec[:, enc == 4] = 1
            Mask = np.eye(D)
            TP = self.evaluate(np.vstack([np.zeros((1, D)), Dec * Mask]))
            TDec.append(Dec), TMask.append(Mask), TPop.append(TP[1:])
            C = cons(TP)
            Fit += nd_sort(np.hstack([objs(TP), C]) if C.size else objs(TP), None, np.inf)[0]
        self.Fitness = Fit
        std0 = Fit[0]
        best_idx = int(np.argmin(Fit[1:]))
        best_score = Fit[1:][best_idx]
        seq_set = [[k] for k in range(1, D + 1)]           # positions in Fitness (0 is the all-zero reference)
        L = 1
        best_seq = None
        while best_score <= std0 * L:
            best_seq = seq_set[best_idx]
            rest = [k for k in range(1, D + 1) if k not in best_seq]
            seq_set = [best_seq + [k] for k in rest]
            if not seq_set:
                break
            score = np.array([Fit[s].sum() for s in seq_set])
            best_idx = int(np.argmin(score))
            best_score = score[best_idx]
            L += 1
        best_vars = np.array(best_seq) - 1
        Mask = np.zeros((N, D))
        for i in range(N):
            if rng.random() < 0.5:
                Mask[i, best_vars] = 1
            else:
                Mask[i, tournament(2, int(np.ceil(rng.random() * D)), Fit[1:], rng=rng)] = 1
        Dec = lo + rng.random((N, D)) * (up - lo)
        Dec[:, enc == 4] = 1
        pop = self.evaluate(Dec * Mask)
        pop, self.Dec, self.Mask, self.front, self.crowd = self._select(
            Population.merge(pop, *TPop), np.vstack([Dec] + TDec), np.vstack([Mask] + TMask), N)
        self.Delta = np.zeros(D)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _select(self, pop, Dec, Mask, N):
        uni = _first_unique(objs(pop))
        pop, Dec, Mask = pop[uni], Dec[uni], Mask[uni]
        N = min(N, len(pop))
        F = objs(pop)
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        f1 = front == 1
        with np.errstate(invalid="ignore", divide="ignore"):
            Fn = (F - F[f1].min(axis=0)) / (F[f1].max(axis=0) - F[f1].min(axis=0))
        last = np.where(front == maxf)[0]
        dele = truncation(Fn[last], len(last) - N + int(nxt.sum()))
        nxt[last[~dele]] = True
        pop, Dec, Mask, front = pop[nxt], Dec[nxt], Mask[nxt], front[nxt]
        return pop, Dec, Mask, front, crowding(objs(pop), front)

    def _ts(self, values):
        if len(values) == 0:
            return None
        return int(tournament(2, 1, values, rng=self.rng)[0])

    def _off_dec(self, pdec, N2):
        if np.any(self.encoding != 4):
            off = ga_half(self.problem, pdec, rng=self.rng)
            off[:, self.encoding == 4] = 1
            return off
        return np.ones((N2, self.D))

    def _op_fitness(self, pdec, pmask):
        rng, Fit = self.rng, self.Fitness
        n = len(pdec)
        h = n // 2
        P1, P2 = pmask[:h], pmask[h:]
        off = P1.copy()
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where((P1[i] > 0) & (P2[i] == 0))[0]
                t = self._ts(-Fit[idx])
                if t is not None:
                    off[i, idx[t]] = 0
            else:
                idx = np.where((P1[i] == 0) & (P2[i] > 0))[0]
                t = self._ts(Fit[idx])
                if t is not None:
                    off[i, idx[t]] = P2[i, idx[t]]
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where(off[i] > 0)[0]
                t = self._ts(-Fit[idx])
                if t is not None:
                    off[i, idx[t]] = 0
            else:
                idx = np.where(off[i] == 0)[0]
                t = self._ts(Fit[idx])
                if t is not None:
                    off[i, idx[t]] = 1
        return self._off_dec(pdec, h), off

    def _op_delta(self, pdec, pmask):
        rng, Delta = self.rng, self.Delta
        n = len(pdec)
        h = n // 2
        P1, P2 = pmask[:h], pmask[h:]
        off = P1.copy()
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where((P1[i] > 0) & (P2[i] == 0))[0]
                t = self._ts(Delta[idx])
                if t is not None:
                    off[i, idx[t]] = 0
            else:
                idx = np.where((P1[i] == 0) & (P2[i] > 0))[0]
                t = self._ts(-Delta[idx])
                if t is not None:
                    off[i, idx[t]] = P2[i, idx[t]]
        Dl = np.minimum(Delta, 1.0)
        for i in range(h):
            idx = np.where(off[i] != Dl)[0]
            if rng.random() < 0.5:
                t = self._ts(-Dl[idx])
                if t is not None:
                    off[i, idx[t]] = 1
            else:
                t = self._ts(Dl[idx])
                if t is not None:
                    off[i, idx[t]] = 0
        return self._off_dec(pdec, h), off

    def step(self):
        N, rng = self.N, self.rng
        pop = self.pop
        mate = tournament(2, 2 * N, self.front, -self.crowd, rng=rng)
        delta = _relief(self, pop, self.front)
        self.Delta = self.Delta + np.abs(delta)
        if np.all(delta == 0):
            off_dec, off_mask = self._op_fitness(self.Dec[mate], self.Mask[mate])
        else:
            off_dec, off_mask = self._op_delta(self.Dec[mate], self.Mask[mate])
        off = self.evaluate(off_dec * off_mask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd = self._select(
            Population.merge(pop, off), np.vstack([self.Dec, off_dec]), np.vstack([self.Mask, off_mask]), N)
