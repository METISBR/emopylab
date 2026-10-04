# emopylab 2026
"""LMOEA-DS (large-scale evolutionary multi-objective optimization assisted by directed sampling).

Reference:
S. Qin, C. Sun, Y. Jin, Y. Tan, and J. Fieldsend. Large-scale evolutionary multi-objective
optimization assisted by directed sampling. IEEE Transactions on Evolutionary Computation, 2021,
25(4): 724-738.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, first_front, nd_sort, objs, kmeans, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'LMOEADS': {'integer', 'large', 'multi', 'real'}}


def _norm_rows(F):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))


def _cosine(A, B):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A @ B.T) / (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :])


def _row_argmax(C):
    """``max(C,[],2)`` of the reference: NaN entries are ignored (an all-NaN row gives NaN at index 0)."""
    Cn = np.where(np.isnan(C), -np.inf, C)
    idx = np.argmax(Cn, axis=1)
    val = C[np.arange(len(C)), idx]
    allnan = np.all(np.isnan(C), axis=1)
    idx[allnan] = 0
    return np.where(allnan, np.nan, np.max(Cn, axis=1)), idx


def _sort_desc(v):
    """Descending sort with NaN first (as the reference's ``sort(...,'descend')``)."""
    return np.argsort(np.where(np.isnan(v), np.inf, -v), kind="stable")


def _representatives(F, RefV):
    """One solution per reference direction: the closest-to-origin solution of the cluster assigned to it, otherwise the
    solution with the largest cosine that has not been chosen yet."""
    F = _norm_rows(F)
    Nr = len(RefV)
    cosine = _cosine(F, RefV)
    _, assoc = _row_argmax(cosine)
    flag = np.zeros(len(F), bool)
    best = np.zeros(Nr, int)
    current = [np.where(assoc == i)[0] for i in range(Nr)]
    for i in range(Nr):
        cur = current[i]
        if len(cur) > 1:
            normf = np.linalg.norm(F[cur], axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                d1 = normf * (F[cur] @ RefV[i]) / np.linalg.norm(RefV[i]) / normf
            b = cur[np.argsort(np.where(np.isnan(d1), np.inf, d1), kind="stable")[0]]
            best[i], flag[b] = b, True
        elif len(cur) == 1:
            best[i], flag[cur[0]] = cur[0], True
    for i in range(Nr):
        if len(current[i]) == 0:
            ind = _sort_desc(cosine[:, i])
            if len(ind) > Nr:
                k = 0
                while flag[ind[k]]:
                    k += 1
                best[i], flag[ind[k]] = ind[k], True
            else:
                best[i] = ind[0]
    return best


def _sbx_children(P1, P2, proC, disC, rng):
    n, D = P1.shape
    mu = rng.random((n, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (disC + 1)), (2 - 2 * mu) ** (-1 / (disC + 1)))
    beta = beta * (-1.0) ** rng.integers(0, 2, (n, D))
    return beta, mu


def _poly_mut(X, lower, upper, site, disM=20.0, rng=None):
    mu = rng.random(X.shape)
    X = np.minimum(np.maximum(X, lower), upper)
    span = upper - lower
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        X[t] = X[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (X[t] - lower[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        X[t] = X[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (upper[t] - X[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return X


def _unique_rows(X):
    return np.unique(X, axis=0)


class LMOEADS(LoopAlgorithm):
    """Each generation samples solutions along rays from the bounds through representative solutions (directed
    sampling), mates the population with those guiding solutions and selects by domination or by decomposition
    depending on how many reference directions are covered."""

    def __init__(self, pop_size: int = 100, nw: int = 10, ns: int = 30, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.nw, self.ns = int(nw), int(ns)

    def start(self):
        self.RefV, _ = uniform_point(self.N, self.M)

    # -- directed sampling ---------------------------------------------------------------------
    def _directed_sampling(self, pop):
        rng, M, D = self.rng, self.M, self.D
        bound = np.eye(M)
        bound[bound == 0] = 10e-7
        lab = kmeans(self.RefV, self.nw, rng)
        centers = np.vstack([self.RefV[lab == k].mean(axis=0) for k in np.unique(lab)])
        direct = np.vstack([bound, centers])
        nw = len(direct)
        best = _representatives(objs(pop), direct)
        BX = decs(pop)[best]
        lo, up = self.lower, self.upper
        norms = np.concatenate([np.linalg.norm(BX - lo, axis=1), np.linalg.norm(BX - up, axis=1)])
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = np.vstack([BX - lo, BX - up]) / norms[:, None]
        interval = np.linalg.norm(up - lo)
        rand = rng.random((self.ns, 2 * nw)) * interval
        lowmat = lo + rand[:, :nw, None] * direction[None, :nw]
        upmat = up + rand[:, nw:, None] * direction[None, nw:]
        X = np.concatenate([lowmat, upmat], axis=1).reshape(-1, D)
        X = np.maximum(np.minimum(X, up), lo)
        samples = self.evaluate(X)
        return samples[first_front(objs(samples))]

    # -- selections ------------------------------------------------------------------------------
    def _assign(self, pop):
        cos = _cosine(_norm_rows(objs(pop)), self.RefV)
        cmax, assoc = _row_argmax(cos)
        return assoc, cmax

    def _domination_selection(self, pop):
        N = self.N
        F = objs(pop)
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        cd = crowding(F, front)
        last = np.where(front == maxf)[0]
        nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
        return pop[nxt]

    def _decomposition_selection(self, pop, assoc, cmax):
        F = _norm_rows(objs(pop))
        sel = []
        for i in np.unique(assoc):
            cur = np.where(assoc == i)[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                fan = cmax[cur] / np.linalg.norm(F[cur], axis=1)
            fan = np.where(np.isnan(fan), -np.inf, fan)
            sel.append(cur[int(np.argmax(fan))])
        return pop[np.array(sel)]

    def _select(self, com):
        assoc, cmax = self._assign(com)
        if len(np.unique(assoc)) < (2 / 3) * self.N:
            return self._domination_selection(com)
        return self._decomposition_selection(com, assoc, cmax)

    # -- variation ---------------------------------------------------------------------------------
    def _ga_once(self, X, guide):
        rng = self.rng
        X = X[rng.permutation(len(X))]
        n, D = X.shape
        P2 = guide[rng.integers(0, len(guide), n)]
        proC, disC = 0.9, 20.0
        beta, _ = _sbx_children(X, P2, proC, disC, rng)
        beta[rng.random((n, D)) < 0.5] = 1
        beta[np.repeat(rng.random((n, 1)) > proC, D, axis=1)] = 1
        plus = np.repeat(rng.random((n, 1)) < 0.5, D, axis=1)
        off = np.where(plus, (X + P2) / 2 + beta * (X - P2) / 2, (X + P2) / 2 - beta * (X - P2) / 2)
        lower, upper = np.tile(self.lower, (n, 1)), np.tile(self.upper, (n, 1))
        return _poly_mut(off, lower, upper, rng.random((n, D)) < 1.0 / D, rng=rng)

    def _ga_twice(self, X):
        rng, N = self.rng, self.N
        X = X[rng.permutation(len(X))]
        n, D = X.shape
        max_off = N if not (N < 1 or N > n) else n
        if n % 2 == 1:
            X = np.vstack([X, X[[0]]])
        proC, proM, disC = 0.9, 1.0 / D, 20.0
        P1, P2 = X[0::2], X[1::2]
        m = len(P1)
        mu = rng.random((m, D))
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (disC + 1)), (2 - 2 * mu) ** (-1 / (disC + 1)))
        beta = beta * (-1.0) ** rng.integers(0, 2, (m, D))
        beta[rng.random((m, D)) > proC] = 1
        off = np.empty((2 * m, D))
        off[0::2] = (P1 + P2) / 2 + beta * (P1 - P2) / 2
        off[1::2] = (P1 + P2) / 2 - beta * (P1 - P2) / 2
        off = off[:max_off]
        lower, upper = np.tile(self.lower, (max_off, 1)), np.tile(self.upper, (max_off, 1))
        k, miu = rng.random((max_off, D)), rng.random((max_off, D))
        span = upper - lower
        with np.errstate(all="ignore"):
            t = (k <= proM) & (miu < 0.5)
            off[t] = off[t] + span[t] * ((2 * miu[t] + (1 - 2 * miu[t]) * (1 - (off[t] - lower[t]) / span[t]) ** 21) ** (1 / 21) - 1)
            t = (k <= proM) & (miu >= 0.5)
            off[t] = off[t] + span[t] * (1 - (2 * (1 - miu[t]) + 2 * (miu[t] - 0.5) * (1 - (upper[t] - off[t]) / span[t]) ** 21) ** (1 / 21))
        return np.minimum(np.maximum(off, lower), upper)

    def step(self):
        pop = self.pop
        guide = self._directed_sampling(pop)
        off = self.evaluate(_unique_rows(self._ga_once(decs(pop), decs(guide))))
        pop = self._select(Population.merge(pop, guide, off))
        off = self.evaluate(_unique_rows(self._ga_twice(decs(pop))))
        self.pop = self._select(Population.merge(pop, off))
