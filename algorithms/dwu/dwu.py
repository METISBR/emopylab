# emopylab 2026
"""DWU (dominance-weighted uniformity multi-objective evolutionary algorithm).

Reference:
G. Moreira and L. Paquete. Guiding under uniformity measure in the decision space. Proceedings of
the IEEE Latin American Conference on Computational Intelligence, 2019.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'DWU': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def info_dominance(F):
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    D = k == 1                                  # D[i, j]: i dominates j
    return D.T @ D.sum(axis=1)                  # sum of the domination counts of the solutions that dominate i


def _dist_dwu(Xd, Xw, Yd, Yw, largest):
    """For every column solution Y(i): the smallest (or largest) weighted distance to the X solutions and its index."""
    d = np.sqrt(np.maximum(((Xd[:, None, :] - Yd[None, :, :]) ** 2).sum(axis=2), 0))
    d = d / (np.abs(Xw[:, None] - Yw[None, :]) + 1)
    if largest:
        idx = np.argmax(d, axis=0)
    else:
        idx = np.argmin(d, axis=0)
    return d[idx, np.arange(d.shape[1])], idx


def replacement_uniformity(pop, N, front, weight):
    Bd, Bw = decs(pop).copy(), weight.astype(float).copy()
    nxt = np.arange(len(pop))
    Rd = np.zeros((N, Bd.shape[1]))
    Rw = np.zeros(N)
    ind = front == front.min()
    if ind.sum() == 1:
        aux_next = list(nxt[ind])
        rest = ~ind
        nfront, nnext = front[rest], nxt[rest]
        ind1 = nfront == nfront.min()
        aux_next += list(nnext[ind1])
    else:
        aux_next = list(nxt[ind])
    aux_next = np.array(aux_next)
    Ad, Aw = Bd[aux_next], Bw[aux_next]
    D, I = _dist_dwu(Ad, Aw, Ad, Aw, True)
    j = int(np.argmax(D))
    Rd[0], Rw[0] = Ad[j], Aw[j]
    Rd[1], Rw[1] = Ad[I[j]], Aw[I[j]]
    picked = [int(aux_next[j]), int(aux_next[I[j]])]
    remain = [nxt[picked[0]], nxt[picked[1]]]
    keep = np.ones(len(nxt), bool)
    keep[picked] = False
    Bd, Bw, nxt = Bd[keep], Bw[keep], nxt[keep]
    ii = 2
    while ii < N:
        D, _ = _dist_dwu(Rd, Rw, Bd, Bw, False)
        j = int(np.argmax(D))
        Rd[ii], Rw[ii] = Bd[j], Bw[j]
        remain.append(nxt[j])
        keep = np.ones(len(nxt), bool)
        keep[j] = False
        Bd, Bw, nxt = Bd[keep], Bw[keep], nxt[keep]
        ii += 1
    return np.array(remain)


def environmental_selection(pop, N):
    C = cons(pop)
    front, _ = nd_sort(objs(pop), C if C.size else None, N)
    w = info_dominance(objs(pop))
    return pop[replacement_uniformity(pop, N, front, w)]


class DWU(LoopAlgorithm):
    """Dominance-weighted uniformity: solutions are weighted by the domination count of their dominators and the
    survivors are chosen greedily to be maximally spread in decision space (distance discounted by the weight
    difference), starting from the two most distant members of the first level."""

    def step(self):
        pop, N, rng = self.pop, self.N, self.rng
        C = cons(pop)
        cv = np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))
        off = self.evaluate(ga(self.problem, decs(pop[tournament(2, N, cv, rng=rng)]), rng=rng))
        self.pop = environmental_selection(Population.merge(pop, off), N)
