# emopylab 2026
"""PRDH (problem reformulation and duplication handling).

Reference:
R. Jiao, B. Xue, and M. Zhang. Solving multiobjective feature selection problems in classification
via problem reformulation and duplication handling. IEEE Transactions on Evolutionary Computation,
2024, 28(4): 846-860.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'PRDH': {'binary', 'multi'}}


def _first_unique_idx(A):
    return np.unique(A, axis=0, return_index=True)[1]


def _hamming(A, b):
    return np.mean(A != b, axis=1)


def _duplication_selection(index, pop, nd_pop):
    """Among solutions with identical objectives keep the non-dominated ones with respect to their (negated) decision
    space distances to the nearest first-front solutions on each objective."""
    NDF, NDX = objs(nd_pop), decs(nd_pop)
    F, X = objs(pop[index]), decs(pop[index])
    c1 = np.where(np.min(np.abs(F[0, 0] - NDF[:, 0])) == np.abs(F[0, 0] - NDF[:, 0]))[0]
    c2 = np.where(np.min(np.abs(F[0, 1] - NDF[:, 1])) == np.abs(F[0, 1] - NDF[:, 1]))[0]
    o1 = np.mean([_hamming(X, NDX[i]) for i in c1], axis=0)
    o2 = []
    for i in c2:
        cols = np.where(NDX[i] == 1)[0]
        o2.append(_hamming(X[:, cols], NDX[i, cols]) if len(cols) else np.zeros(len(X)))
    o2 = np.mean(o2, axis=0)
    front, _ = nd_sort(-np.column_stack([o1, o2]), None, np.inf)
    return index[front == 1]


def _min_obj1_index(F):
    m = 0
    for i in range(len(F)):
        if F[i, 0] < F[m, 0] or (F[i, 0] == F[m, 0] and F[i, 1] < F[m, 1]):
            m = i
    return m


def _env_selection(pop, N):
    pop = pop[_first_unique_idx(decs(pop))]
    front, _ = nd_sort(objs(pop), None, np.inf)
    nxt = front == 1
    nd_pop = pop[front == 1]
    for i in range(2, int(front[np.isfinite(front)].max()) + 1):
        No = np.where(front == i)[0]
        F = objs(pop[No])
        _, first, c = np.unique(F, axis=0, return_index=True, return_inverse=True)
        c = np.asarray(c).reshape(-1)
        order = np.argsort(np.argsort(first, kind="stable"), kind="stable")   # 'stable' group numbering by first appearance
        c = order[c]
        chosen = []
        for j in range(c.max() + 1):
            idx = np.where(c == j)[0]
            chosen.extend(_duplication_selection(No[idx], pop, nd_pop) if len(idx) > 1 else No[idx])
        nxt[np.array(chosen, dtype=int)] = True
    pop, front = pop[nxt], front[nxt]
    F = objs(pop)
    min2 = F[_min_obj1_index(F), 1]
    if np.sum(F[:, 1] <= min2) >= N:
        keep = F[:, 1] <= min2
        pop, front = pop[keep], front[keep]
        cum = 0
        for max_f in range(1, int(front.max()) + 1):
            cum += np.sum(front == max_f)
            if cum >= N:
                break
        nxt = front < max_f
        cd = crowding(objs(pop), front)
        last = np.where(front == max_f)[0]
        nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
        return pop[nxt], front[nxt], cd[nxt]
    rank = np.argsort(F[:, 0], kind="stable")
    N = min(N, len(pop))
    return pop[rank[:N]], np.arange(1, N + 1, dtype=float), np.zeros(N)


class PRDH(LoopAlgorithm):
    """Bi-objective binary search that removes duplicated decision vectors, keeps objective-duplicates that are
    far from the first front in decision space, and only ranks solutions not worse than the lightest one on f2."""

    def _initialize_infill(self):
        rng, N, D = self.rng, self.N, self.D
        T = min(D, N * 3)
        X = np.zeros((N, D))
        for i in range(N):
            k = int(rng.integers(1, T + 1))
            X[i, rng.permutation(D)[:k]] = 1
        return self.evaluate(X)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        _, self.front, self.crowd = _env_selection(infills, self.N)   # the selected set only provides the ranks
        self._set_optimum()

    def _reproduce(self, parents):
        """Uniform crossover + bit-flip mutation of binary parents, duplicates and all-zero vectors removed."""
        if np.any(self.encoding != 4):
            raise ValueError("PRDH supports binary decision variables only")
        rng = self.rng
        P = decs(parents)
        h = len(P) // 2
        P1, P2 = P[:h], P[h: 2 * h]
        D = P.shape[1]
        k = rng.random((h, D)) < 0.5
        O1, O2 = P1.copy(), P2.copy()
        O1[k], O2[k] = P2[k], P1[k]
        O = np.vstack([O1, O2])
        site = rng.random((2 * h, D)) < 1.0 / D
        O[site] = 1 - O[site]
        zero = O.sum(axis=1) == 0
        if zero.any():
            O[zero] = rng.integers(0, 2, (int(zero.sum()), D))
        O = np.unique(O, axis=0)
        O = O[O.sum(axis=1) > 0]
        return self.evaluate(O)

    def step(self):
        pop, N = self.pop, self.N
        mate = tournament(2, N, self.front, -self.crowd, rng=self.rng)
        off = self._reproduce(pop[mate])
        self.pop, self.front, self.crowd = _env_selection(Population.merge(pop, off), N)
