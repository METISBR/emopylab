# emopylab 2026
"""tDEA-CPBI (theta-dominance based evolutionary algorithm with CPBI).

Reference:
F. Ming, W. Gong, L. Wang, and L. Gao. A constraint-handling technique for decomposition-based
constrained many-objective evolutionary algorithms. IEEE Transactions on Systems, Man, and
Cybernetics: Systems, 2023, 53(12): 7783-7793.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, cons, cosine_distance, decs, first_index_reaching, ga, nd_sort,
                                            objs, uniform_point)
from core.population import Population

ALGORITHM_FLAGS = {'tDEACPBI': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _cons(pop):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), 1))


def _normalization(F, C, z, znad, zc, znadc):
    N, M = F.shape
    zc = np.minimum(zc, C.min(axis=0))
    z = np.minimum(z, F.min(axis=0))
    znadc = np.maximum(znadc, C.max(axis=0))
    W = np.full((M, M), 1e-6)
    np.fill_diagonal(W, 1.0)
    with np.errstate(all="ignore"):
        asf = np.stack([np.max(np.abs((F - z) / (znad - z)) / W[i], axis=1) for i in range(M)], axis=1)
        extreme = np.argmin(asf, axis=0)
        try:
            a = 1.0 / np.linalg.solve(F[extreme] - z, np.ones(M)) + z
        except np.linalg.LinAlgError:
            a = np.full(M, np.nan)
    if np.any(np.isnan(a)) or np.any(a <= z):
        a = F.max(axis=0)
    znad = a
    with np.errstate(all="ignore"):
        Cn = np.nan_to_num((C - zc) / (znadc - zc))
        Fn = (F - z) / (znad - z)
    return Fn, Cn, z, znad, zc, znadc


def _t_ncd_sort(F, C, W, fr):
    N, NW = len(F), len(W)
    Fn, Cn, *_ = _normalization(F, C, F.min(axis=0), F.max(axis=0), C.min(axis=0), C.max(axis=0))
    norm_p = np.linalg.norm(Fn, axis=1)
    cosine = 1.0 - cosine_distance(Fn, W)
    d1 = norm_p[:, None] * cosine
    d2 = norm_p[:, None] * np.sqrt(np.maximum(0.0, 1 - cosine ** 2))
    d3 = np.sum(np.maximum(0, Cn), axis=1)
    cls = np.argmin(d2, axis=1)
    theta = np.full(NW, 5.0)
    theta[np.sum(W > 1e-4, axis=1) == 1] = 1e6
    t_front = np.zeros(N)
    for i in range(NW):
        Cidx = np.where(cls == i)[0]
        # the reference implementation adds the scalar (1-fr)*d3(i) (indexing d3 by the reference-vector id)
        key = d1[Cidx, i] + theta[i] * d2[Cidx, i] + (1 - fr) * (d3[i] if i < N else 0.0)
        t_front[Cidx[np.argsort(key, kind="stable")]] = np.arange(1, len(Cidx) + 1)
    return t_front


def _environmental_selection(pop, W, N, z, znad, zc, znadc, rng):
    F, C = objs(pop), _cons(pop)
    front_no, max_f = nd_sort(F, C, N)
    St = np.where(front_no <= max_f)[0]
    Fn, Cn, z, znad, zc, znadc = _normalization(F[St], C[St], z, znad, zc, znadc)
    fr = np.sum(np.sum(np.maximum(0, Cn), axis=1) == 0) / N
    t_front = _t_ncd_sort(Fn, Cn, W, fr)
    max_t = first_index_reaching(t_front, N)
    last = np.where(t_front == max_t)[0]
    last = last[rng.permutation(len(last))]
    t_front = t_front.copy()
    t_front[last[: int(np.sum(t_front <= max_t)) - N]] = np.inf
    return pop[St[t_front <= max_t]], z, znad, zc, znadc


class tDEACPBI(LoopAlgorithm):
    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        F, C = objs(self.pop), _cons(self.pop)
        self.z, self.znad, self.zc, self.znadc = F.min(axis=0), F.max(axis=0), C.min(axis=0), C.max(axis=0)

    def step(self):
        pool = self.rng.integers(0, len(self.pop), size=self.N)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.z, self.znad, self.zc, self.znadc = _environmental_selection(
            Population.merge(self.pop, off), self.W, self.N, self.z, self.znad, self.zc, self.znadc, self.rng)
