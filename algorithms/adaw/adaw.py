# emopylab 2026
"""AdaW: evolutionary algorithm with adaptive weights.

Reference:
M. Li and X. Yao. What weights work for you? Adapting weights for any
Pareto front shape in decomposition-based evolutionary multiobjective
optimisation. Evolutionary Computation, 2020, 28(2): 227-253.

A MOEA/D-style search whose weight set is periodically updated from an
external archive: "undeveloped" archive members (far from the population)
spawn new weights, and poorly performing weights are deleted, so the weight
distribution follows any Pareto-front shape.

Backend: emopylab array facade; MLX selected automatically by
``core.algorithm.Algorithm`` on Apple Silicon when available.
"""

from __future__ import annotations

import numpy as np
from core.population import Population
from util.optimum import filter_optimum

from algorithms.community_utils.base import LoopAlgorithm
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGAhalf import OperatorGAhalf
from operators.utility_functions.UniformPoint import UniformPoint
from util.array_backend import backend_cdist


ALGORITHM_FLAGS = {"AdaW": {"multi", "many"}}

_EPS = 1e-12


def _tche(F: np.ndarray, Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AdaW Tchebycheff: max |f - z| / w (division by the weight)."""
    return np.max(np.abs(F - Z[None, :]) / np.maximum(W, _EPS), axis=1)


def _niche_radius_matrix(objs: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalized distance matrix and niche radius r."""
    n, M = objs.shape
    f_min, f_max = objs.min(axis=0), objs.max(axis=0)
    norm = (objs - f_min[None, :]) / np.maximum(f_max - f_min, _EPS)[None, :]
    d = backend_cdist(norm, norm)
    np.fill_diagonal(d, np.inf)
    sd = np.sort(d, axis=1)
    col = min(M, sd.shape[1]) - 1
    r = float(np.median(sd[:, col]))
    return d, max(r, _EPS)


def archive_update(archive: Population, n_max: int) -> Population:
    """Niche-based archive truncation."""
    if len(archive) <= n_max:
        return archive
    objs = np.asarray(archive.get("F"), dtype=float)
    d, r = _niche_radius_matrix(objs)
    R = np.minimum(d / r, 1.0)
    keep = np.ones(len(archive), dtype=bool)
    while keep.sum() > n_max:
        idx = np.where(keep)[0]
        crowd = 1.0 - np.prod(R[np.ix_(idx, idx)], axis=1)
        worst = idx[int(np.argmax(crowd))]
        keep[worst] = False
    return archive[keep]


class AdaW(LoopAlgorithm):
    """AdaW (Li & Yao, 2020): steady-state MOEA/D (one GAhalf offspring per weight, replacing the first improved
    neighbour) with a niche-truncated archive of non-dominated solutions and periodic weight adaptation."""

    def __init__(self, pop_size: int = 100, adapt_weights: bool = True, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.adapt_weights = bool(adapt_weights)
        self.ext_archive: Population = Population.empty()

    def initial_size(self) -> int:
        W, n_eff = UniformPoint(self.pop_size, int(self.problem.n_obj))
        self.W = np.asarray(W, dtype=float)
        self.pop_size = int(max(1, n_eff))
        self.T = int(np.ceil(self.pop_size / 10))
        self._update_neighbors()
        return self.pop_size

    def _update_neighbors(self):
        d = backend_cdist(self.W, self.W)
        self.B = np.argsort(d, axis=1, kind="stable")[:, : self.T]

    def start(self):
        F = np.asarray(self.pop.get("F"), dtype=float)
        self.Z = F.min(axis=0)
        fn, _ = NDSort(F, 1)
        self.ext_archive = self.pop[np.asarray(fn).reshape(-1) == 1]

    def step(self):
        rng, N = self.rng, self.pop_size
        offs = []
        for i in range(N):
            P = self.B[i][rng.permutation(self.B.shape[1])] if rng.random() < 0.9 else rng.permutation(N)
            X = np.asarray(self.pop.get("X"), dtype=float)
            off = self.evaluate(np.asarray(OperatorGAhalf(self.problem, X[P[:2]], rng=rng), dtype=float))
            offs.append(off)
            f = np.asarray(off.get("F"), dtype=float)[0]
            self.Z = np.minimum(self.Z, f)
            F_pop = np.asarray(self.pop.get("F"), dtype=float)
            g_old = _tche(F_pop[P], self.Z, self.W[P])
            g_new = _tche(np.tile(f, (len(P), 1)), self.Z, self.W[P])
            better = np.where(g_old >= g_new)[0]
            if better.size:
                self.pop[int(P[better[0]])] = off[0]
        merged = Population.merge(self.ext_archive, *offs)
        fn, _ = NDSort(np.asarray(merged.get("F"), dtype=float), 1)
        self.ext_archive = archive_update(merged[np.asarray(fn).reshape(-1) == 1], 2 * N)
        if self.adapt_weights:
            period = int(np.ceil(0.05 * int(np.ceil(self.max_FE / N))))
            if int(np.ceil(self.FE / N)) % max(1, period) == 0 and self.FE <= 0.9 * self.max_FE:
                self._weight_update()

    # -- weight adaptation ------------------------------------------------
    def _weight_update(self):
        archive = self.ext_archive
        if len(archive) == 0 or len(self.pop) == 0:
            return
        arc_obj = np.asarray(archive.get("F"), dtype=float)
        pop_obj = np.asarray(self.pop.get("F"), dtype=float)
        M = arc_obj.shape[1]

        f_min = arc_obj.min(axis=0)
        f_max = arc_obj.max(axis=0)
        denom = np.maximum(f_max - f_min, _EPS)
        arc_n = (arc_obj - f_min[None, :]) / denom[None, :]
        pop_n = (pop_obj - f_min[None, :]) / denom[None, :]

        dis1 = np.sort(backend_cdist(arc_n, pop_n), axis=1)
        dis2 = np.sort(backend_cdist(arc_n, arc_n), axis=1)
        col = 1 if dis2.shape[1] >= 2 else 0
        niche = float(np.median(dis2[:, col]))

        und_mask = dis1[:, 0] >= niche
        und = archive[und_mask]
        if len(und):
            und_obj = np.asarray(und.get("F"), dtype=float)
            denom_w = np.maximum(
                und_obj.sum(axis=1) - float(np.sum(self.Z)), _EPS,
            )[:, None]
            W1 = (und_obj - self.Z[None, :]) / denom_w
            for i in range(len(W1)):
                W_all = np.vstack([self.W, W1[i][None, :]])
                d = backend_cdist(W_all, W_all)
                np.fill_diagonal(d, np.inf)
                B1 = np.argsort(d, axis=1)[:, : self.T]

                pop1 = Population.merge(self.pop, und[[i]])
                F1 = np.asarray(pop1.get("F"), dtype=float)
                nb = B1[-1]
                g_all = _tche(F1[nb], self.Z, np.tile(W1[i], (len(nb), 1)))
                g_new = float(_tche(und_obj[i][None, :], self.Z, W1[i][None, :])[0])
                if not np.any(g_all < g_new):
                    self.W = np.vstack([self.W, W1[i][None, :]])
                    self.pop = Population.merge(self.pop, und[[i]])
                    F_pop = np.asarray(self.pop.get("F"), dtype=float)
                    P = nb[nb < len(self.pop)]
                    g_old = _tche(F_pop[P], self.Z, self.W[P])
                    g_rep = _tche(
                        np.repeat(und_obj[i][None, :], len(P), axis=0), self.Z, self.W[P],
                    )
                    for j in P[g_old > g_rep]:
                        self.pop[int(j)] = und[i]

        # Delete poorly performed weights back to N.
        while len(self.pop) > self.pop_size:
            objs = np.asarray(self.pop.get("F"), dtype=float)
            _, ia, bi = np.unique(objs, axis=0, return_index=True, return_inverse=True)
            if len(ia) == len(bi):
                d, r = _niche_radius_matrix(objs)
                R = np.minimum(d / r, 1.0)
                keep = np.ones(len(self.pop), dtype=bool)
                while keep.sum() > self.pop_size:
                    idx = np.where(keep)[0]
                    crowd = 1.0 - np.prod(R[np.ix_(idx, idx)], axis=1)
                    keep[idx[int(np.argmax(crowd))]] = False
                self.pop = self.pop[keep]
                self.W = self.W[keep]
            else:
                counts = np.bincount(bi)
                mode_val = int(np.argmax(counts))
                index = np.where(bi == mode_val)[0]
                g = _tche(objs[index], self.Z, self.W[index])
                worst = index[int(np.argmax(g))]
                keep = np.ones(len(self.pop), dtype=bool)
                keep[worst] = False
                self.pop = self.pop[keep]
                self.W = self.W[keep]

        self._update_neighbors()

    def _set_optimum(self):
        # AdaW reports the archive (non-dominated) as the optimum set.
        base = self.ext_archive if len(self.ext_archive) else self.pop
        self.opt = filter_optimum(base, least_infeasible=True)


ALGORITHMS = {"AdaW": AdaW}
