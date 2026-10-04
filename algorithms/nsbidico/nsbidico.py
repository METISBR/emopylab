# emopylab 2026
"""NSBiDiCo (non-dominated sorting bidirectional differential coevolution algorithm).

Reference:
C. S. R. Mendes, A. F. R. Araujo, and L. R. C. Farias. Non-dominated sorting bidirectional
differential coevolution. Proceedings of the IEEE International Conference on Systems, Mans and
Cybernetics, 2023.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, cv, cosine_distance, de, decs, first_front, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'NSBiDiCo': {'constrained', 'integer', 'multi', 'real'}}


def _environmental_selection(pop, N):
    F, c = objs(pop), cons(pop)
    front_no, max_f = nd_sort(F, c if c.size else None, N)
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt]


def _last_selection(F, con, K, zmax, rng):
    N = len(F)
    with np.errstate(all="ignore"):
        P = (F - zmax) / (F.min(axis=0) - zmax - 1e-10)
    P = np.nan_to_num(P)
    P[P == 0] = 1e-10
    cosine = (1.0 - cosine_distance(P)) * (1 - np.eye(N))
    dele = np.zeros(N, bool)
    while dele.sum() < K:
        rows, cols = np.where(cosine == cosine.max())
        j = int(rng.integers(0, len(rows)))
        a, b = int(rows[j]), int(cols[j])
        drop = a if (con[a] < con[b] or (con[a] == con[b] and rng.random() < 0.5)) else b
        dele[drop] = True
        cosine[:, drop] = 0
        cosine[drop, :] = 0
    return dele


def _update_arc(pop, N, rng):
    F, c = objs(pop), cv(pop)
    front1 = first_front(np.column_stack([F, c]))
    pop = pop[front1]
    pop = pop[cv(pop) > 0]
    if len(pop) < N:
        return pop
    dele = _last_selection(objs(pop), -cv(pop), len(pop) - N, objs(pop).max(axis=0), rng)
    return pop[~dele]


class NSBiDiCo(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, Cr: float = 1, F: float = 0.5, proM: float = 1, disM: float = 20, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.params = (float(Cr), float(F), float(proM), float(disM))

    def start(self):
        self.arc = self.pop[:0]

    def _mating_selection(self):
        rng, N, pop, arc = self.rng, self.N, self.pop, self.arc
        if len(arc) < N:
            return pop[tournament(2, N, -cv(pop), rng=rng)]
        allp = Population.merge(pop, arc)
        F = objs(allp)
        with np.errstate(all="ignore"):
            P = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-10) + 1e-10
        cosine = (1.0 - cosine_distance(P)) * (1 - np.eye(len(P)))
        temp = np.sort(-cosine, axis=1)
        rank = np.lexsort(temp.T[::-1])                     # sortrows: row indices in lexicographic order
        cv1, cv2 = cv(pop), cv(arc)
        ang1, ang2 = rank[:N], rank[N:]
        i1, i2 = rng.integers(0, N, size=N + 2), rng.integers(0, len(arc), size=N + 2)
        pool, i = [], 0
        while len(pool) < N:
            pool.append(pop[int(i1[i])] if cv1[i1[i]] < cv2[i2[i]] else arc[int(i2[i])])
            pool.append(pop[int(i1[i + 1])] if ang1[i1[i + 1]] < ang2[i2[i + 1]] else arc[int(i2[i + 1])])
            i += 2
        return Population.create(pool[:N])

    def step(self):
        rng, N = self.rng, self.N
        all_pop = Population.merge(self.pop, self.arc) if len(self.arc) else self.pop
        pool = self._mating_selection()
        off = self.evaluate(de(self.problem, decs(pool), decs(pool[rng.integers(0, N, size=N)]), decs(pool[rng.integers(0, N, size=N)]), self.params, rng=rng))
        self.arc = _update_arc(Population.merge(all_pop, off), N, rng)
        self.pop = _environmental_selection(Population.merge(self.pop, off), N)
