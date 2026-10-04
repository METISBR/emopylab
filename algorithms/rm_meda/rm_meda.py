# emopylab 2026
"""RM-MEDA (regularity model-based multiobjective estimation of distribution).

Reference:
Q. Zhang, A. Zhou, and Y. Jin. RM-MEDA: A regularity model-based multiobjective estimation of
distribution algorithm. IEEE Transactions on Evolutionary Computation, 2008, 12(1): 41-63.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'RMMEDA': {'integer', 'multi', 'real'}}


def _local_pca(X, M, K, rng):
    N, D = X.shape
    means = [X[k].copy() for k in range(K)]
    PI = [np.eye(D) for _ in range(K)]
    evec, eval_ = [None] * K, [None] * K
    partition = np.zeros(N, dtype=int)
    for _ in range(50):
        dist = np.stack([np.sum(((X - means[k]) @ PI[k]) * (X - means[k]), axis=1) for k in range(K)], axis=1)
        partition = np.argmin(dist, axis=1)
        updated = np.zeros(K, bool)
        for k in range(K):
            old = means[k]
            cur = np.where(partition == k)[0]
            if len(cur) < 2:
                if len(cur) == 0:
                    cur = np.array([int(rng.integers(0, N))])
                means[k], PI[k], evec[k], eval_[k] = X[cur[0]].copy(), np.eye(D), None, None
                updated[k] = bool(np.linalg.norm(old - means[k]) > 1e-5) or len(cur) == 0
            else:
                means[k] = X[cur].mean(axis=0)
                w, V = np.linalg.eigh(np.atleast_2d(np.cov(X[cur] - means[k], rowvar=False)))
                order = np.argsort(-w, kind="stable")
                eval_[k], evec[k] = w[order], V[:, order]
                PI[k] = evec[k][:, M - 1:] @ evec[k][:, M - 1:].T
                updated[k] = bool(np.linalg.norm(old - means[k]) > 1e-5)
        if not updated.any():
            break
    a = np.zeros((K, M - 1))
    b = np.zeros((K, M - 1))
    for k in range(K):
        if evec[k] is not None:
            h = (X[partition == k] - means[k]) @ evec[k][:, : M - 1]
            a[k], b[k] = h.min(axis=0), h.max(axis=0)
    vol = np.prod(b - a, axis=1)
    prob = np.cumsum(vol / vol.sum()) if vol.sum() > 0 else np.cumsum(np.full(K, 1.0 / K))
    return means, evec, eval_, a, b, prob


class RMMEDA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, K: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.K = int(K)

    def _operator(self):
        rng, M, D = self.rng, self.M, self.D
        X = decs(self.pop)
        N = len(X)
        means, evec, eval_, a, b, prob = _local_pca(X, M, self.K, rng)
        off = np.zeros((N, D))
        for i in range(N):
            k = int(np.argmax(rng.random() <= prob))
            if evec[k] is not None:
                lo, up = a[k] - 0.25 * (b[k] - a[k]), b[k] + 0.25 * (b[k] - a[k])
                trial = rng.random(M - 1) * (up - lo) + lo
                sigma = np.sum(np.abs(eval_[k][M - 1:D])) / (D - M + 1)
                off[i] = means[k] + trial @ evec[k][:, : M - 1].T + rng.standard_normal(D) * np.sqrt(sigma)
            else:
                off[i] = means[k] + rng.standard_normal(D)
        return self.evaluate(off)

    def step(self):
        off = self._operator()
        pop = Population.merge(self.pop, off)
        F = objs(pop)
        front_no, max_f = nd_sort(F, None, self.N)
        nxt = front_no < max_f
        last = np.where(front_no == max_f)[0]
        while len(last) > self.N - int(nxt.sum()):
            last = np.delete(last, int(np.argmin(crowding(F[last]))))
        nxt[last] = True
        self.pop = pop[nxt]
