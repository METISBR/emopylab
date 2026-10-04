# emopylab 2026
"""MOEA-D-M2M (mOEA/D based on MOP to MOP).

Reference:
H. Liu, F. Gu, and Q. Zhang. Decomposition of a multiobjective optimization problem into a number of
simple multiobjective subproblems. IEEE Transactions on Evolutionary Computation, 2014, 18(3):
450-455.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, crowding, decs, nd_sort, objs, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MOEADM2M': {'integer', 'multi', 'real'}}


def _associate(pop, W, S, rng):
    K = len(W)
    F = objs(pop)
    transformation = np.argmin(cosine_distance(F, W), axis=1)             # max cosine similarity
    partition = np.zeros((S, K), dtype=int)
    for i in range(K):
        current = np.where(transformation == i)[0]
        if len(current) < S:
            current = np.concatenate([current, rng.integers(0, len(pop), size=S - len(current))])
        elif len(current) > S:
            front_no, max_f = nd_sort(F[current], None, S)
            front_no = front_no.copy()
            last = np.where(front_no == max_f)[0]
            rank = np.argsort(crowding(F[current[last]]), kind="stable")
            front_no[last[rank[: int(np.sum(front_no <= max_f)) - S]]] = np.inf
            current = current[front_no <= max_f]
        partition[:, i] = current
    return pop[partition.reshape(-1, order="F")]


class MOEADM2M(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, K: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.K = int(K)

    def initial_size(self):
        self.W, self.K = uniform_point(self.K, self.M)
        self.pop_size = int(np.ceil(self.pop_size / self.K) * self.K)
        self.S = self.pop_size // self.K
        return self.pop_size

    def start(self):
        self.pop = _associate(self.pop, self.W, self.S, self.rng)

    def _operator(self, P1, P2):
        rng = self.rng
        N, D = P1.shape
        lo, up = self.lower, self.upper
        expo = -(1.0 - self.FE / self.max_FE) ** 0.7
        with np.errstate(all="ignore"):
            rc = (2 * rng.random((N, 1)) - 1) * (1 - rng.random((N, 1)) ** expo)
            off = P1 + rc * (P1 - P2)
            rm = 0.25 * (2 * rng.random((N, D)) - 1) * (1 - rng.random((N, D)) ** expo)
        site = rng.random((N, D)) < 1.0 / D
        off[site] = off[site] + (rm * (up - lo))[site]
        below, above = off < lo, off > up
        rnd = rng.random((N, D))
        lo_b, up_b = np.broadcast_to(lo, (N, D)), np.broadcast_to(up, (N, D))
        off[below] = lo_b[below] + 0.5 * rnd[below] * (P1 - lo_b)[below]
        off[above] = up_b[above] - 0.5 * rnd[above] * (up_b - P1)[above]
        return self.evaluate(off)

    def step(self):
        S, K, N, rng = self.S, self.K, self.N, self.rng
        local = rng.integers(0, S, size=(S, K)) + np.arange(K) * S
        glob = rng.integers(0, N, size=(S, K))
        local = np.where(rng.random((S, K)) < 0.7, glob, local)
        mate = self.pop[local.reshape(-1, order="F")]
        off = self._operator(decs(self.pop), decs(mate))
        self.pop = _associate(Population.merge(self.pop, off), self.W, S, rng)
