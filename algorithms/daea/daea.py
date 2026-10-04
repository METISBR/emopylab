# emopylab 2026
"""DAEA (duplication analysis based evolutionary algorithm).

Reference:
H. Xu, B. Xue, and M. Zhang. A duplication analysis based evolutionary algorithm for bi-objective
feature selection. IEEE Transactions on Evolutionary Computation, 2021, 25(2): 205-218.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'DAEA': {'binary', 'multi'}}


def _unique_rows_first(X):
    _, idx = np.unique(X, axis=0, return_index=True)
    return idx                                  # sorted by row value, first occurrence


def environmental_selection(pop, N, rng):
    idx = _unique_rows_first(decs(pop))
    UP = pop[idx]
    if len(UP) > N:
        X, F = decs(UP), objs(UP)
        D = X.shape[1]
        SD = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
        np.fill_diagonal(SD, np.inf)
        _, inv = np.unique(F, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        dup = []
        for i in range(inv.max() + 1):
            j = np.where(inv == i)[0]
            if len(j) > 1:
                t = X[j[0]].sum()
                d = SD[np.ix_(j, j)].min(axis=1) / 2
                with np.errstate(all="ignore"):
                    p = d / t
                    r = np.where(p < 0.8 - 0.6 * (t - 1) / (D - 1))[0]
                if len(r):
                    dup += list(j[r[rng.permutation(len(r))[: len(r) - 1]]])
        if len(UP) - len(dup) > N:
            UP = UP[np.delete(np.arange(len(UP)), np.array(dup, dtype=int))]
        F = objs(UP)
        front, maxf = nd_sort(F, None, N)
        sel = front < maxf
        cand = front == maxf
        cd = crowding(F, front)
        while sel.sum() < N:
            S = F[sel, 0]
            ic = np.where(cand)[0]
            ic = ic[np.argsort(-cd[ic], kind="stable")]
            cnt = np.array([np.sum(S == F[k, 0]) for k in ic])
            pick = ic[int(np.argsort(cnt, kind="stable")[0])]
            sel[pick] = True
            cand[pick] = False
        return UP[sel]
    return Population.merge(UP, pop[rng.permutation(len(pop))[: N - len(UP)]])


class DAEA(LoopAlgorithm):
    """Decomposition-free binary EA for sparse combinatorial problems: niche-based mating in objective space,
    exchange of a random subset of differing bits, sparsity-preserving bit-flip mutation and a selection that
    removes near-duplicate objective vectors with similar decision vectors before crowding-based truncation."""

    def _initialize_infill(self):
        N, D, rng = self.N, self.D, self.rng
        T = min(D, N * 3)
        if T < D:
            X = np.zeros((N, D))
            for i in range(N):
                k = int(rng.integers(1, T + 1))
                X[i, rng.permutation(D)[:k]] = 1
            return self.evaluate(X)
        return self.evaluate(self.random_decs(N))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _variation(self):
        pop, rng = self.pop, self.rng
        F, X = objs(pop), decs(pop) != 0
        N, D = X.shape
        T = max(4, int(np.ceil(N * 0.2)))
        with np.errstate(all="ignore"):
            nf = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        ed = np.sqrt(np.maximum(((nf[:, None, :] - nf[None, :, :]) ** 2).sum(axis=2), 0))
        np.fill_diagonal(ed, np.inf)
        nic = np.argsort(np.where(np.isnan(ed), np.inf, ed), axis=1, kind="stable")[:, :T]
        ip2 = np.zeros(N, dtype=int)
        for i in range(N):
            if rng.random() < 0.8:
                ip2[i] = nic[i, int(rng.integers(T))]
            else:
                g = np.delete(np.arange(N), i)
                ip2[i] = g[int(rng.integers(N - 1))]
        P1, P2 = X, X[ip2]
        off = P1.copy()
        for i in range(N):
            k = np.where(P1[i] ^ P2[i])[0]
            t = len(k)
            if t > 1:
                j = k[rng.permutation(t)[: int(rng.integers(1, t))]]
                off[i, j] = P2[i, j]
        for i in range(N):
            if rng.random() < 0.2:
                j1, j0 = np.where(off[i])[0], np.where(~off[i])[0]
                k1 = rng.random(len(j1)) < 1.0 / (len(j1) + 1)
                k0 = rng.random(len(j0)) < 1.0 / (len(j0) + 1)
                off[i, j1[k1]] = False
                off[i, j0[k0]] = True
            else:
                k = rng.random(D) < 1.0 / D
                off[i, k] = ~off[i, k]
        off = np.unique(off, axis=0)
        return self.evaluate(off.astype(float))

    def step(self):
        off = self._variation()
        self.pop = environmental_selection(Population.merge(self.pop, off), self.N, self.rng)
