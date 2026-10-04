# emopylab 2026
"""RSEA (radial space division based evolutionary algorithm).

Reference:
C. He, Y. Tian, Y. Jin, X. Zhang, and L. Pan. A radial space division based evolutionary algorithm
for many-objective optimization. Applied Soft Computing, 2017, 61: 603-621.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, first_front, ga, nd_sort, objs, pdist2, tournament
from core.population import Population

ALGORITHM_FLAGS = {'RSEA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _radar_grid(P, div):
    N, M = P.shape
    theta = np.arange(M) * 2 * np.pi / M
    with np.errstate(all="ignore"):
        rloc = np.column_stack([np.sum(P * np.cos(theta), axis=1) / P.sum(axis=1), np.sum(P * np.sin(theta), axis=1) / P.sum(axis=1)])
    rloc = np.nan_to_num((rloc + 1) / 2)
    yl, yu = rloc.min(axis=0), rloc.max(axis=0)
    nrloc = rloc if np.any(yu == yl) else (rloc - yl) / (yu - yl)
    gloc = np.floor(nrloc * div)
    gloc[gloc >= div] = div - 1
    uniq, site = np.unique(gloc, axis=0, return_inverse=True)
    return site.reshape(-1), rloc


def _last_selection(algo, P, choose, N, div):
    M = P.shape[1]
    with np.errstate(all="ignore"):
        pbi = np.linalg.norm(P, axis=1)[:, None] * np.sqrt(np.maximum(0.0, 1 - (1 - cosine_distance(P, np.eye(M))) ** 2))
    extreme = np.argmin(pbi, axis=0)
    choose = choose.copy()
    choose[extreme] = True
    con = np.linalg.norm(P, axis=1)
    con = con / con.max()
    site, rloc = _radar_grid(P, div)
    rdis = pdist2(rloc, rloc)
    np.fill_diagonal(rdis, np.inf)
    crowd = np.bincount(site[choose], minlength=site.max() + 1).astype(float)
    while choose.sum() < N:
        remain_s = np.where(~choose)[0]
        remain_g = np.unique(site[remain_s])
        best_g = remain_g[crowd[remain_g] == crowd[remain_g].min()]
        current = remain_s[np.isin(site[remain_s], best_g)]
        r = 1 - (algo.FE / algo.max_FE) ** 2
        fitness = algo.M * r * con[current] - rdis[np.ix_(current, np.where(choose)[0])].min(axis=1)
        b = current[int(np.argmin(fitness))]
        choose[b] = True
        crowd[site[b]] += 1
    return choose


def _environmental_selection(algo, pop, rng_, N):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = np.where(front_no <= max_f)[0]
    lo, up = rng_[0], rng_[1]
    P = F[nxt] if np.any(lo == up) else (F[nxt] - lo) / (up - lo)
    choose = _last_selection(algo, P, np.isin(nxt, np.where(front_no < max_f)[0]), N, int(np.ceil(np.sqrt(N))))
    pop = pop[nxt[choose]]
    F = objs(pop)
    rng_[0] = np.minimum(rng_[0], F.min(axis=0))
    rng_[1] = F[first_front(F)].max(axis=0)
    return pop


def _mating_selection(F, rng_, N, rng):
    P = (F - rng_[0]) / (rng_[1] - rng_[0])
    con = np.linalg.norm(P, axis=1)
    site, _ = _radar_grid(P, int(np.ceil(np.sqrt(len(P)))))
    crowd = np.bincount(site).astype(float)
    n = int(np.ceil(N / 2) * 2)
    grids = tournament(2, n, crowd, rng=rng)
    pool = np.zeros(n, dtype=int)
    for i in range(n):
        cur = np.where(site == grids[i])[0]
        if len(cur) == 0:
            pool[i] = int(rng.integers(0, len(P)))
        else:
            par = cur[rng.integers(0, len(cur), size=2)]
            pool[i] = par[int(np.argmin(con[par]))]
    return pool


class RSEA(LoopAlgorithm):
    def start(self):
        self.range = np.full((2, self.M), np.inf)

    def step(self):
        F = objs(self.pop)
        self.range[0] = np.minimum(self.range[0], F.min(axis=0))
        self.range[1] = F[first_front(F)].max(axis=0)
        pool = _mating_selection(F, self.range, self.N, self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(self, Population.merge(self.pop, off), self.range, self.N)
