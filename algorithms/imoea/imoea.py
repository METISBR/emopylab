# emopylab 2026
"""Indicator-driven MOEA with adaptive weight vectors (Han and Li, DOCS 2025).

This module ports the IMOEA search scheme: a MOEA/D-Tchebycheff evolution
loop whose weight vectors are periodically reshaped by a WS-transformation.
An external archive tracks elite trade-off solutions, a sparsity score
singles out overcrowded and underexplored subregions, and a normalized R2
indicator decides when the weight set has grown stale and must adapt.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from algorithms.community_utils.moead_family import (
    ensure_population,
    ind_F,
    pop_F,
    rng_from_algo,
    sample_initial,
    set_optimum_from_pop,
    tchebycheff_values,
    weight_vectors,
    neighbors,
)
from core.algorithm import Algorithm
from core.population import Population
from operators.crossover.sbx import SBX
from operators.mutation.pm import PolynomialMutation
from util.optimum import filter_optimum

ALGORITHM_FLAGS = {"IMOEA": {"multi", "many", "real"}}

_EPS = 1e-12


def _threshold_for_dim(n_obj: int) -> float:
    """Stagnation tolerance TH(M) governing weight adaptation eagerness."""
    m = int(n_obj)
    if m >= 5:
        return 0.05 * (1 + (m - 5))
    return 0.1 * (1 + (m - 5))


def _r2_value(W: np.ndarray, F: np.ndarray, z: np.ndarray) -> float:
    """Normalized R2 indicator of archive F under weight set W (lower is better)."""
    F = np.asarray(F, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    lo = np.min(F, axis=0)
    hi = np.max(F, axis=0)
    span = np.maximum(hi - lo, _EPS)
    Fn = (F - lo) / span
    zn = np.clip((z - lo) / span, -1.0, 2.0)
    vals = np.max(np.abs(Fn[:, None, :] - zn[None, None, :]) * W[None, :, :], axis=2)
    return float(np.mean(np.min(vals, axis=0)))


def _niche_crowding(F: np.ndarray, k: int = 3) -> np.ndarray:
    """Niche crowding radius D(p): median distance to the k nearest neighbors.

    Objectives are min-max normalized first so every dimension contributes
    evenly to the radius. Large values mark isolated solutions.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n <= 1:
        return np.full(n, np.inf)
    lo, hi = np.min(F, axis=0), np.max(F, axis=0)
    N = (F - lo) / np.maximum(hi - lo, _EPS)
    kk = max(1, min(int(k), n - 1))
    D = np.empty(n)
    for i in range(n):
        d = np.linalg.norm(N - N[i], axis=1)
        d[i] = np.inf
        D[i] = float(np.median(np.partition(d, kk - 1)[:kk]))
    return D


def _sparsity_volumes(F: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Subspace sparsity V: enclosed volume between each solution and the
    component-wise maximum of its neighborhood mates (neighbor-max point)."""
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    V = np.empty(n)
    for i in range(n):
        pool = np.asarray(B[i], dtype=int)
        pool = pool[pool != i]
        ref = np.max(F[pool], axis=0) if pool.size else F[i] + 1.0
        V[i] = float(np.prod(np.maximum(ref - F[i], 0.0) + _EPS))
    s = np.sum(V)
    return V / s if s > _EPS else np.full(n, 1.0 / max(n, 1))


class IMOEA(Algorithm):
    """Indicator-driven MOEA with online weight-vector adaptation.

    The population evolves under Tchebycheff scalarization on a simplex
    weight set. An external archive (capped at twice the population size)
    preserves elite solutions; when the normalized R2 improvement of the
    archive stagnates inside the active window, the sparsest subregion
    receives a fresh weight while an overcrowded one is released.
    """

    def __init__(
        self,
        pop_size: int = 100,
        delta: float = 0.9,
        nr: int = 2,
        nus: Any = 0.1,
        sampling: Any = None,
        crossover: Any = None,
        mutation: Any = None,
        sbx_eta: float = 20.0,
        pm_eta: float = 20.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self.pop_size = int(max(2, pop_size))
        self.delta = float(delta)
        self.nr = int(max(1, nr))
        self.nus = nus
        self.sampling = sampling
        self.crossover = crossover or SBX(prob=0.9, eta=float(sbx_eta), n_offsprings=1)
        self.mutation = mutation or PolynomialMutation(eta=float(pm_eta))
        self._cooldown = 5
        self._last_adapt = -10**9

    # -- helpers ---------------------------------------------------------
    def _max_gen(self) -> int:
        term = getattr(self, "termination", None)
        for attr in ("n_max_gen", "max_gen"):
            v = getattr(term, attr, None)
            if v is not None:
                try:
                    return max(1, int(v))
                except Exception:
                    pass
        for attr in ("n_max_evals", "n_max_eval", "max_evals"):
            v = getattr(term, attr, None)
            if v is not None:
                try:
                    return max(1, int(v) // max(1, self.pop_size))
                except Exception:
                    pass
        return 100

    def _nus_threshold(self, progress: float) -> float:
        spec = self.nus
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            a, b = float(spec[0]), float(spec[1])
            return a + (b - a) * float(np.clip(progress, 0.0, 1.0))
        return float(spec)

    def _truncate_archive(self, arch: Population, cap: int) -> Population:
        if arch is None or len(arch) <= cap:
            return arch
        F = pop_F(arch)
        keep = set()
        for m in range(F.shape[1]):
            keep.add(int(np.argmin(F[:, m])))
        if len(keep) < cap:
            D = _niche_crowding(F)
            order = np.argsort(-D, kind="stable")
            for idx in order:
                if int(idx) not in keep:
                    keep.add(int(idx))
                if len(keep) >= cap:
                    break
        return arch[np.asarray(sorted(keep), dtype=int)]

    def _refresh_archive(self, candidates: Population) -> None:
        cap = 2 * self.pop_size
        merged = Population.merge(self.EA, candidates) if self.EA is not None else candidates
        nd = filter_optimum(merged, least_infeasible=True)
        if not isinstance(nd, Population):
            nd = merged
        self.EA = self._truncate_archive(nd, cap)

    # -- lifecycle -------------------------------------------------------
    def _initialize_infill(self) -> Population:
        self.W, n = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = int(n)
        self.T = int(max(2, min(self.pop_size, int(np.ceil(self.pop_size / 10)))))
        self.B = neighbors(self.W, self.T)
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        self.pop = infills
        F = pop_F(self.pop)
        self.Z = np.min(F, axis=0)
        self.EA = self._truncate_archive(
            filter_optimum(self.pop, least_infeasible=True), 2 * self.pop_size
        )
        if not isinstance(self.EA, Population):
            self.EA = self.pop
        r2 = _r2_value(self.W, pop_F(self.EA), self.Z)
        self._r2_init, self._r2_prev = r2, r2

    def _infill(self) -> Population:
        rng = rng_from_algo(self)
        pairs = []
        self._pools = []
        for i in range(self.pop_size):
            if rng.random() < self.delta:
                pool = np.asarray(self.B[i], dtype=int)
            else:
                pool = np.arange(self.pop_size)
            self._pools.append(pool)
            p = rng.choice(pool, size=2, replace=len(pool) < 2)
            pairs.append([self.pop[int(p[0])], self.pop[int(p[1])]])
        off = self.crossover.do(self.problem, pairs, random_state=rng)
        off = self.mutation.do(self.problem, off, random_state=rng)
        return ensure_population([off])

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            return
        off_F = pop_F(infills)
        self.Z = np.minimum(self.Z, np.min(off_F, axis=0))
        pools = getattr(self, "_pools", None)
        for j in range(min(len(infills), self.pop_size)):
            off = infills[j]
            fj = off_F[j] if off_F.ndim == 2 else np.asarray(ind_F(off), dtype=float)
            pool = np.asarray(pools[j], dtype=int) if pools is not None else np.arange(self.pop_size)
            g_old = tchebycheff_values(pop_F(self.pop[pool]), self.Z, self.W[pool])
            g_new = tchebycheff_values(np.repeat(fj[None, :], len(pool), axis=0), self.Z, self.W[pool])
            repl = np.where(g_old >= g_new)[0][: self.nr]
            for r in repl:
                self.pop[int(pool[r])] = off
        self._refresh_archive(infills)
        self._maybe_adapt_weights()

    def _maybe_adapt_weights(self) -> None:
        max_gen = self._max_gen()
        gen = int(self.n_gen or 0)
        progress = gen / max(max_gen, 1)
        if progress < 0.2 or progress > 0.9 or self.EA is None or len(self.EA) == 0:
            return
        if gen - int(self._last_adapt) < self._cooldown:
            r2 = _r2_value(self.W, pop_F(self.EA), self.Z)
            self._r2_prev = r2
            return
        EA_F = pop_F(self.EA)
        r2 = _r2_value(self.W, EA_F, self.Z)
        improvement = (float(self._r2_prev) - r2) / max(abs(float(self._r2_prev)), _EPS)
        self._r2_prev = r2
        if improvement >= _threshold_for_dim(self.problem.n_obj):
            return
        V = _sparsity_volumes(pop_F(self.pop), self.B)
        s = int(np.argmax(V))
        d = int(np.argmin(V))
        if s == d or V[d] >= self._nus_threshold(progress):
            return
        dist = np.linalg.norm(self.W - self.W[s], axis=1)
        dist[s] = -np.inf
        f = int(np.argmax(dist))
        w_new = (self.W[s] + self.W[f]) / 2.0
        w_new = np.clip(w_new, 0.0, None)
        tot = float(np.sum(w_new))
        w_new = w_new / tot if tot > _EPS else np.full_like(w_new, 1.0 / len(w_new))
        self.W[d] = np.maximum(w_new, _EPS)
        self.W[d] /= float(np.sum(self.W[d]))
        g = tchebycheff_values(EA_F, self.Z, np.repeat(self.W[d][None, :], len(EA_F), axis=0))
        self.pop[d] = self.EA[int(np.argmin(g))]
        self.Z = np.minimum(self.Z, np.asarray(ind_F(self.pop[d]), dtype=float))
        self.B = neighbors(self.W, self.T)
        self._last_adapt = gen

    def _set_optimum(self) -> None:
        set_optimum_from_pop(self)


ALGORITHMS = {"IMOEA": IMOEA}
