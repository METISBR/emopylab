# emopylab 2026
"""HypE (hypervolume estimation algorithm).

Reference:
J. Bader and E. Zitzler. HypE: An algorithm for fast hypervolume-based many-objective optimization.
Evolutionary Computation, 2011, 19(1): 45-76.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'HypE': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _hypesub(l, A, M, bounds, pvec, alpha, k):
    """Exact hypervolume contribution by slicing (recursive over the objectives)."""
    h = np.zeros(l)
    order = np.argsort(A[:, M - 1], kind="stable")
    S, pvec = A[order], pvec[order]
    for i in range(len(S)):
        extrusion = (S[i + 1, M - 1] if i < len(S) - 1 else bounds[M - 1]) - S[i, M - 1]
        if M == 1:
            if i + 1 > k:
                break
            if np.all(alpha >= 0):
                h[pvec[: i + 1]] += extrusion * alpha[i]
        elif extrusion > 0:
            h = h + extrusion * _hypesub(l, S[: i + 1], M - 1, bounds, pvec[: i + 1], alpha, k)
    return h


def cal_hv(points, bounds, k, n_sample, rng):
    """HypE fitness: exact slicing for two objectives, Monte-Carlo (``n_sample`` points) otherwise."""
    N, M = points.shape
    k = int(k)
    if M > 2:
        alpha = np.zeros(N)
        for i in range(1, k + 1):
            j = np.arange(1, i)
            alpha[i - 1] = np.prod((k - j) / (N - j)) / i
        fmin = points.min(axis=0)
        S = rng.uniform(fmin, bounds, size=(n_sample, M))
        dom = points[:, 0:1] <= S[None, :, 0]                                      # (N, nS), one objective at a time
        for m in range(1, M):
            dom &= points[:, m:m + 1] <= S[None, :, m]
        dS = dom.sum(axis=0)
        table = np.concatenate([[0.0], alpha])
        F = (dom * table[dS][None, :]).sum(axis=1)
        return F * np.prod(bounds - fmin) / n_sample
    alpha = np.zeros(k)
    for i in range(1, k + 1):
        j = np.arange(1, i)
        alpha[i - 1] = np.prod((k - j) / (N - j)) / i
    return _hypesub(N, points, M, bounds, np.arange(N), alpha, k)


def _environmental_selection(pop, N, ref, n_sample, rng):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    choose = np.ones(len(last), bool)
    while choose.sum() > N - int(nxt.sum()):
        remain = np.where(choose)[0]
        f = cal_hv(F[last[remain]], ref, choose.sum() - N + int(nxt.sum()), n_sample, rng)
        choose[remain[int(np.argmin(f))]] = False
    nxt[last[choose]] = True
    return pop[nxt]


class HypE(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, nSample: int = 10000, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.n_sample = int(nSample)

    def start(self):
        self.ref = np.zeros(self.M) + objs(self.pop).max(axis=0) * 1.2

    def step(self):
        fit = cal_hv(objs(self.pop), self.ref, self.N, self.n_sample, self.rng)
        pool = tournament(2, self.N, -fit, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N, self.ref, self.n_sample, self.rng)
