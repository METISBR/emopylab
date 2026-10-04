# emopylab 2026
"""hpaEA (hyperplane assisted evolutionary algorithm).

Reference:
H. Chen, Y. Tian, W. Pedrycz, G. Wu, R. Wang, and L. Wang. Hyperplane assisted evolutionary
algorithm for many-objective optimization problems. IEEE Transactions on Cybernetics, 2020, 50(7):
3367-3380.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, angle_matrix, decs, ga, nd_sort, objs, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'hpaEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _farthest_fill(ang, nxt, N):
    while nxt.sum() < N:
        sel, rem = np.where(nxt)[0], np.where(~nxt)[0]
        nxt[rem[int(np.argmax(ang[np.ix_(rem, sel)].min(axis=1)))]] = True


def _nd_selection(pop, V, N, fe, max_fe):
    F = objs(pop)
    span = F.max(axis=0) - F.min(axis=0)
    with np.errstate(all="ignore"):
        Fn = (F - F.min(axis=0)) / span
    n, M = Fn.shape
    nxt = np.zeros(n, bool)
    IM = np.argsort(Fn, axis=0, kind="stable")                       # IM[:, m]: solutions sorted by objective m
    pos = np.empty_like(IM)
    for m in range(M):
        pos[IM[:, m], m] = np.arange(n)                              # position of every solution in column m
    for i in range(n):
        if np.all(pos[i] != 0):                                      # not the minimum of any objective
            neigh = np.array([IM[pos[i, m] - 1, m] for m in range(M)])
            if len(np.unique(neigh)) < len(neigh) or np.any(neigh < 2):   # indices < 3 in 1-based terms
                continue
            A = np.ones((M, M))
            A[:, :-1] = Fn[neigh, :-1]
            B = Fn[neigh, -1]
            if np.linalg.det(A) < 1e-10:
                continue
            coef = np.linalg.solve(A, B)
            if Fn[i, -1] <= np.sum(coef[:-1] * Fn[i, :-1]) + coef[-1]:
                nxt[i] = True
    osi = nxt.copy()
    ang_v = angle_matrix(Fn, V)
    assoc = np.argmin(ang_v, axis=1)
    for v in np.unique(assoc):
        cur = np.where(assoc == v)[0]
        nxt[cur[int(np.argmin(ang_v[cur, v]))]] = True
    if fe > 0.8 * max_fe:
        n_each = int(np.floor((2 / 3) * (N / M)))
        for m in range(M):
            col = Fn[:, m]
            interval = (col.max() - col.min()) / n_each
            with np.errstate(all="ignore"):
                idx = np.ceil((col - col.min() + 1e-5) / interval)
            for f in np.unique(idx):
                cur = np.where(idx == f)[0]
                if not np.any(nxt[cur]) or f == 1:
                    nxt[cur[0]] = True
    _farthest_fill(angle_matrix(Fn), nxt, N)
    psi = np.cumsum(nxt)[np.where(nxt & osi)[0]] - 1                  # positions inside the selection
    return pop[nxt], psi


def _environmental_selection(pop, V, N, fe, max_fe):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    sel = front_no <= max_f
    if sel.sum() <= N:
        psi = np.cumsum(sel)[np.where(sel & (front_no == 1))[0]] - 1
        return pop[sel], psi
    if max_f <= 1:
        return _nd_selection(pop[front_no == 1], V, N, fe, max_fe)
    sel_s, can = np.where(front_no < max_f)[0], np.where(front_no == max_f)[0]
    P = np.vstack([F[sel_s], F[can]])
    with np.errstate(all="ignore"):
        Pn = (P - P.min(axis=0)) / (P.max(axis=0) - P.min(axis=0))
    merged = pop[np.concatenate([sel_s, can])]
    nxt = np.concatenate([np.ones(len(sel_s), bool), np.zeros(len(can), bool)])
    _farthest_fill(angle_matrix(np.nan_to_num(Pn)), nxt, N)
    return merged[nxt], np.arange(len(sel_s))


class hpaEA(LoopAlgorithm):
    def initial_size(self):
        self.V, _ = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        self.max_obj = np.full(self.M, np.inf)
        self.psi = np.zeros(0, dtype=int)

    def step(self):
        rng, N = self.rng, self.N
        pool = np.concatenate([rng.integers(0, len(self.pop), size=N - len(self.psi)), self.psi]).astype(int)
        pool = pool[rng.permutation(len(pool))]
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=rng))
        comb = Population.merge(self.pop, off)
        keep = np.max(objs(comb) - self.max_obj, axis=1) <= 0
        cand = comb[keep] if keep.sum() >= 2 else comb
        temp_max = objs(cand).max(axis=0)
        rep = self.max_obj > temp_max
        self.max_obj[rep] = temp_max[rep]
        self.pop, self.psi = _environmental_selection(cand, self.V, N, self.FE, self.max_FE)
