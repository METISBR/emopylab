# emopylab 2026
"""c-DPEA (constrained dual-population evolutionary algorithm).

Reference:
M. Ming, A. Trivedi, R. Wang, and D. Srinivasan. A dual-population based evolutionary algorithm for
constrained multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2021,
25(4): 739-753.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, objs, pdist2, tournament, truncate_lexi, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'cDPEA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _cos_region(F, W):
    """Index of the closest weight vector by angle (a zero vector maps to the first one)."""
    with np.errstate(all="ignore"):
        d = 1 - (F @ W.T) / (np.linalg.norm(F, axis=1)[:, None] * np.linalg.norm(W, axis=1)[None, :])
    return np.argmin(np.where(np.isnan(d), np.inf, d), axis=1)


def _truncation(F, K):
    d = pdist2(F, F)
    np.fill_diagonal(d, np.inf)
    return truncate_lexi(d, K)


def _kth_distance(F):
    n = len(F)
    d = pdist2(F, F)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, max(int(np.floor(np.sqrt(n))) - 1, 0)]


def _convergence_rank(F, front, crowd, N):
    n = len(F)
    nxt = front == 1
    middle = np.zeros(n)
    if nxt.sum() > N:
        idx = np.where(nxt)[0]
        dele = idx[_truncation(F[nxt], int(nxt.sum()) - N)]
        middle[dele] = 1
        order = np.lexsort((-crowd, middle, front))
    else:
        order = np.lexsort((-crowd, front))
    rank = np.zeros(n)
    rank[order] = np.arange(1, n + 1)
    return rank


def _finish(pop, rank_conv, front_d, crowd, alpha, N):
    n = len(pop)
    order = np.lexsort((-crowd, front_d))
    rank_div = np.zeros(n)
    rank_div[order] = np.arange(1, n + 1)
    rank = alpha * rank_conv + (1 - alpha) * rank_div
    return pop[np.argsort(rank, kind="stable")[:N]], np.arange(1, N + 1)


def _region_ranks(PF, F, W, region, z):
    front_d = np.ones(len(F))
    for i in range(len(W)):
        idx = np.where(region == i)[0]
        if len(idx):
            g = np.sum((PF[idx] - z) * W[i], axis=1)
            front_d[idx[np.argsort(g, kind="stable")]] = np.arange(1, len(idx) + 1)
    return front_d


def env_selection(pop, N, alpha):
    F, C = objs(pop), cons(pop)
    n = len(pop)
    z, zmax = F.min(axis=0), F.max(axis=0)
    W, _ = uniform_point(N, len(z))
    region = _cos_region(F - z, W)
    PF = F.copy()
    infe = np.any(C > 0, axis=1) if C.size else np.zeros(n, bool)
    if infe.any():
        r = _cos_region((zmax - z)[None, :], W)[0]
        PF[infe] = zmax + np.sum(np.maximum(0, C[infe]), axis=1)[:, None] * W[r] / np.linalg.norm(W[r])
    CV = np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(n)
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    Dom = (CV[:, None] < CV[None, :]) | ((CV[:, None] == CV[None, :]) & (k == 1))
    front = Dom.sum(axis=1) @ Dom + 1
    crowd = _kth_distance(F)
    rank_conv = _convergence_rank(F, front, crowd, N)
    front_d = _region_ranks(PF, F, W, region, z) + infe * n
    return _finish(pop, rank_conv, front_d, crowd, alpha, N)


def env_selection_no_con(pop, N, alpha, gamma, para):
    F, C = objs(pop), cons(pop)
    n = len(pop)
    z = F.min(axis=0)
    infe = np.any(C > 0, axis=1) if C.size else np.zeros(n, bool)
    cvs = np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(n)
    phi_max = cvs[infe].max() if infe.any() else 0.0
    W, _ = uniform_point(N, len(z))
    region = _cos_region(F - z, W)
    PF = F.copy()
    for i in range(len(W)):
        idx = np.where(region == i)[0]
        bad = infe[idx]
        if len(idx) and bad.any():
            fmax = F[idx].max(axis=0)
            with np.errstate(all="ignore"):
                w = (cvs[idx[bad]] / phi_max) ** (np.exp(para) / max(gamma, 0.000001))
            PF[idx[bad]] = F[idx[bad]] + w[:, None] * (fmax - F[idx[bad]])
    k = (PF[:, None, :] < PF[None, :, :]).any(axis=2).astype(int) - (PF[:, None, :] > PF[None, :, :]).any(axis=2).astype(int)
    Dom = k == 1
    front = Dom.sum(axis=1) @ Dom + 1
    crowd = _kth_distance(F)
    rank_conv = _convergence_rank(F, front, crowd, N)
    front_d = _region_ranks(PF, F, W, region, z)
    return _finish(pop, rank_conv, front_d, crowd, alpha, N)


class cDPEA(LoopAlgorithm):
    """Constrained dual-population evolutionary algorithm: one population respects the constraints, the other
    ignores them with a repair-like shift of the infeasible solutions; both rank their members by a mix of
    convergence and diversity and mate together."""

    def __init__(self, pop_size: int = 100, sampling=None, dual_population=True, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.dual_population = dual_population

    def _alpha(self):
        return 2.0 / (1 + np.exp(-self.FE * 10 / self.max_FE)) - 1

    def _para(self):
        return np.ceil(self.max_FE / self.N) / 2 - np.ceil(self.FE / self.N)

    def start(self):
        self.P1 = self.pop
        self.P2 = self.pop
        self.alpha, self.para = self._alpha(), self._para()

    def step(self):
        N, rng = self.N, self.rng
        self.P1 = self.P1[rng.permutation(N)]
        self.P2 = self.P2[rng.permutation(N)]
        F1 = {tuple(r) for r in objs(self.P1)}
        lia = np.array([tuple(r) in F1 for r in objs(self.P2)])
        gamma = 1 - lia.sum() / N
        self.P1, r1 = env_selection(self.P1, N, self.alpha)
        self.P2, r2 = env_selection_no_con(self.P2, N, self.alpha, gamma, self.para)
        allp = Population.merge(self.P1, self.P2)
        pool = tournament(2, 2 * N, np.concatenate([r1, r2]), rng=rng)
        off = self.evaluate(ga_half(self.problem, decs(allp[pool]), rng=rng))
        self.alpha, self.para = self._alpha(), self._para()
        self.P1, _ = env_selection(Population.merge(self.P1, off), N, self.alpha)
        self.P2, _ = env_selection_no_con(Population.merge(self.P2, off), N, self.alpha, gamma, self.para)
        self.pop = self.P1
