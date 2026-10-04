# emopylab 2026
"""BiCo (bidirectional coevolution constrained multiobjective evolutionary algorithm).

Reference:
Z. Liu, B. Wang, and K. Tang. Handling constrained multiobjective optimization problems via
bidirectional coevolution. IEEE Transactions on Cybernetics, 2022, 52(10): 10163-10176.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, pdist2, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'BiCo': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _cv(pop):
    C = cons(pop)
    return np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))


def _cos_matrix(P):
    n = np.linalg.norm(P, axis=1)
    with np.errstate(all="ignore"):
        C = (P @ P.T) / (n[:, None] * n[None, :])
    C = np.nan_to_num(C)
    return C * (1 - np.eye(len(P)))


def _normalized(F, extra=1e-10):
    zmin = F.min(axis=0)
    return (F - zmin) / (F.max(axis=0) - zmin + extra) + extra


def env_selection(pop, N):
    C = cons(pop)
    front, maxf = nd_sort(objs(pop), C if C.size else None, N)
    nxt = front < maxf
    last = np.where(front == maxf)[0]
    K = int(nxt.sum()) + len(last) - N
    if K == 0:
        nxt[last] = True
    else:
        P = _normalized(objs(pop))[last]
        d = pdist2(P, P)
        np.fill_diagonal(d, np.inf)
        dele = truncate_lexi(d, K)
        nxt[last[~dele]] = True
    return pop[nxt]


def _mating_selection(pop, arc, N, rng):
    if arc is None or len(arc) < N:
        return pop[tournament(2, N, -_cv(pop), rng=rng)]
    allp = Population.merge(pop, arc)
    P = _normalized(objs(allp))
    cosine = _cos_matrix(P)
    temp = np.sort(-cosine, axis=1)
    rank = np.lexsort(temp.T[::-1])
    cv1, cv2 = _cv(pop), _cv(arc)
    ang1, ang2 = rank[:N], rank[N:len(allp)]
    i1 = rng.integers(0, N, N)
    i2 = rng.integers(0, len(arc), N)
    picks = []
    i = 0
    while len(picks) < N:
        j = i % N
        picks.append(pop[[i1[j]]] if cv1[i1[j]] < cv2[i2[j]] else arc[[i2[j]]])
        j = (i + 1) % N
        picks.append(pop[[i1[j]]] if ang1[i1[j]] < ang2[i2[j]] else arc[[i2[j]]])
        i += 2
    return Population.merge(*picks)


def update_arc(pop, N, rng):
    cv = _cv(pop)
    F = objs(pop)
    front, _ = nd_sort(np.column_stack([F, cv]), None, 1)
    pop = pop[front == 1]
    pop = pop[_cv(pop) > 0]
    if len(pop) < N:
        return pop
    F = objs(pop)
    cvp = _cv(pop)
    K = len(pop) - N
    with np.errstate(all="ignore"):
        Fn = (F - F.max(axis=0)) / (F.min(axis=0) - F.max(axis=0) - 1e-10)
    cos = _cos_matrix(Fn)
    dele = np.zeros(len(pop), bool)
    while dele.sum() < K:
        cols, rows = np.where(cos.T == cos.max())
        j = int(rng.integers(len(rows)))
        t1, t2 = rows[j], cols[j]
        if cvp[t1] > cvp[t2] or (cvp[t1] == cvp[t2] and rng.random() < 0.5):
            v = t1
        else:
            v = t2
        dele[v] = True
        cos[:, v] = 0
        cos[v, :] = 0
    return pop[~dele]


class BiCo(LoopAlgorithm):
    """Bidirectional coevolution for constrained problems: the main population is selected by constrained
    non-domination while an archive of non-dominated infeasible solutions (diversity kept by angle-based
    truncation) supplies mating partners from the other side of the constraint boundary."""

    def start(self):
        self.arc = None

    def step(self):
        N, rng = self.N, self.rng
        pop = self.pop
        allp = pop if self.arc is None else Population.merge(pop, self.arc)
        pool = _mating_selection(pop, self.arc, N, rng)
        off = self.evaluate(ga(self.problem, decs(pool[np.arange(N)]), rng=rng))
        self.arc = update_arc(Population.merge(allp, off), N, rng)
        self.pop = env_selection(Population.merge(pop, off), N)
