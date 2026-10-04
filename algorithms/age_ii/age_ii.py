# emopylab 2026
"""AGE-II (approximation-guided evolutionary multi-objective algorithm II).

Reference:
M. Wagner and F. Neumann. A fast approximation-guided evolutionary multi-objective algorithm.
Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2013, 687-694.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, first_front, ga, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'AGEII': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _update_archive(pop, epsilon):
    grid = np.floor(objs(pop) / epsilon)
    return pop[first_front(grid)]


def _mating_selection(F, rng):
    front_no, _ = nd_sort(F, None, np.inf)
    cd = crowding(F, front_no)
    remain = np.where(rng.random(len(F)) < 1.0 / front_no)[0]
    return remain[tournament(2, len(F), -cd[remain], rng=rng)]


def _environmental_selection(pop, arc, N):
    F = objs(pop)
    NP, NA = len(pop), len(arc)
    discard = np.zeros(NP, bool)
    for i in range(N, NP):
        discard[i] = np.any(np.all(F[i] >= arc + 1, axis=1))
    pop, F = pop[~discard], F[~discard]
    NP = len(pop)
    alpha = np.max(F[:, None, :] - arc[None, :, :], axis=2)            # (NP, NA)
    rank = np.argsort(alpha, axis=0, kind="stable")                    # per archive member
    rho = np.take_along_axis(alpha, rank, axis=0)
    remain = np.arange(NP)
    while len(remain) > N:
        S = np.tile(rho[0], (len(remain), 1))
        best = rank[0][None, :] == remain[:, None]
        S[best] = np.broadcast_to(rho[1], S.shape)[best]
        worst = np.lexsort(np.sort(S, axis=1)[:, ::-1].T[::-1])[0]
        keep = rank != remain[worst]
        L = len(remain)
        rho = rho.T[keep.T].reshape(NA, L - 1).T
        rank = rank.T[keep.T].reshape(NA, L - 1).T
        remain = np.delete(remain, worst)
    return pop[remain]


class AGEII(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, epsilon: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.epsilon = float(epsilon)

    def start(self):
        self.archive = _update_archive(self.pop, self.epsilon)

    def step(self):
        pool = _mating_selection(objs(self.pop), self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.archive = _update_archive(Population.merge(self.archive, off), self.epsilon)
        self.pop = _environmental_selection(Population.merge(self.pop, off), objs(self.archive), self.N)
