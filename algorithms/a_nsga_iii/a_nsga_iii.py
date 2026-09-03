# emopylab 2026
"""A-NSGA-III: Adaptive NSGA-III.

Faithful port of the PlatEMO implementation.

Reference:
H. Jain and K. Deb. An evolutionary many-objective optimization algorithm
using reference-point based non-dominated sorting approach, part II:
Handling constraints and extending to an adaptive approach. IEEE
Transactions on Evolutionary Computation, 2014, 18(4): 602-622.

Reference points are relocated online: around each crowded point (rho >= 2)
a simplex of M new points at spacing ``interval`` is inserted, and inserted
points with no associated solution are removed.  Classic adaptive baseline
for irregular Pareto fronts.

Backend: emopylab array facade; MLX selected automatically by
``core.algorithm.Algorithm`` on Apple Silicon when available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from core.population import Population
from util.optimum import filter_optimum

from core.algorithm import Algorithm
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection
from operators.utility_functions.UniformPoint import UniformPoint

from algorithms.community_utils.moead_family import sample_initial

ALGORITHM_FLAGS = {"A-NSGA-III": {"multi", "many"}, "ANSGAIII": {"multi", "many"}}

_EPS = 1e-12


def _perp_distance(P: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Perpendicular distances from points P (rows) to reference lines Z."""
    norm_p = np.linalg.norm(P, axis=1, keepdims=True)
    nz = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), _EPS)
    cos = np.clip((P @ Z.T) / (np.maximum(norm_p, _EPS) * nz.T), -1.0, 1.0)
    return norm_p * np.sqrt(np.maximum(1.0 - cos ** 2, 0.0))


def _associate_rho(P: np.ndarray, Z: np.ndarray) -> np.ndarray:
    dist = _perp_distance(P, Z)
    pi = np.argmin(dist, axis=1)
    return np.bincount(pi, minlength=len(Z))


def adaptive_reference(
    pop_obj: np.ndarray, Z: np.ndarray, N: int, interval: float,
) -> np.ndarray:
    """Addition and deletion of reference points (PlatEMO Adaptive.m)."""
    M = pop_obj.shape[1]
    rho = _associate_rho(pop_obj, Z)
    old_Z = None
    while np.any(rho >= 2) and (old_Z is None or not np.array_equal(old_Z, Z)):
        old_Z = Z.copy()
        for i in np.where(rho >= 2)[0]:
            p = np.tile(Z[i], (M, 1)) - interval / M
            p[np.eye(M, dtype=bool)] += interval
            Z = np.vstack([Z, p])
        Z = Z[~np.any(Z < 0.0, axis=1)]
        _, index = np.unique(np.round(Z * 1e4) / 1e4, axis=0, return_index=True)
        Z = Z[np.sort(index)]
        rho = _associate_rho(pop_obj, Z)
    # Deletion: added points (index >= N) with no associated solution.
    to_delete = np.intersect1d(np.arange(N, len(Z)), np.where(rho == 0)[0])
    if to_delete.size:
        keep = np.ones(len(Z), dtype=bool)
        keep[to_delete] = False
        Z = Z[keep]
    return Z


def environmental_selection(
    pop: Population, N: int, Z: np.ndarray, z_min: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Population:
    """NSGA-III environmental selection (PlatEMO EnvironmentalSelection.m)."""
    F = np.asarray(pop.get("F"), dtype=float)
    if z_min is None:
        z_min = np.ones(Z.shape[1])
    fn, max_fno = NDSort(F, N)
    fn = np.asarray(fn, dtype=float).reshape(-1)
    next_mask = fn < float(max_fno)
    last = np.where(fn == float(max_fno))[0]
    k = int(N - next_mask.sum())
    choose = _last_selection(F[next_mask], F[last], k, Z, z_min, rng)
    next_mask[last[choose]] = True
    return pop[next_mask]


def _last_selection(
    F1: np.ndarray, F2: np.ndarray, K: int, Z: np.ndarray, z_min: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    P = (np.vstack([F1, F2]) if F1.size else np.asarray(F2, dtype=float)) - z_min[None, :]
    N, M = P.shape
    n1, n2 = len(F1), len(F2)
    NZ = len(Z)

    # Adaptive normalization via extreme points and hyperplane intercepts.
    w = np.full((M, M), 1e-6) + np.eye(M)
    extreme = np.zeros(M, dtype=int)
    for i in range(M):
        extreme[i] = int(np.argmin(np.max(P / w[i][None, :], axis=1)))
    try:
        hyperplane = np.linalg.solve(P[extreme], np.ones(M))
        a = 1.0 / hyperplane
        if np.any(~np.isfinite(a)) or np.any(a <= 0):
            a = P.max(axis=0)
    except np.linalg.LinAlgError:
        a = P.max(axis=0)
    Pn = P / np.maximum(a, _EPS)[None, :]

    dist = _perp_distance(Pn, Z)
    pi = np.argmin(dist, axis=1)
    d = dist[np.arange(N), pi]
    rho = np.bincount(pi[:n1], minlength=NZ)

    choose = np.zeros(n2, dtype=bool)
    z_choose = np.ones(NZ, dtype=bool)
    while choose.sum() < K:
        cand = np.where(z_choose)[0]
        if cand.size == 0:
            rest = np.where(~choose)[0]
            take = rng.permutation(rest)[: K - int(choose.sum())]
            choose[take] = True
            break
        cand = cand[rho[cand] == rho[cand].min()]
        j = int(rng.choice(cand))
        I = np.where(~choose & (pi[n1:] == j))[0]
        if I.size:
            if rho[j] == 0:
                s = I[np.argmin(d[n1 + I])]
            else:
                s = rng.choice(I)
            choose[int(s)] = True
            rho[j] += 1
        else:
            z_choose[j] = False
    return np.where(choose)[0]


class ANSGAIII(Algorithm):
    """Adaptive NSGA-III (Jain & Deb, 2014)."""

    def __init__(
        self,
        pop_size: int = 100,
        sampling=None,
        seed: Optional[int] = None,
        use_gpu: bool = False,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(
            seed=seed, use_gpu=use_gpu, array_backend=array_backend,
            gpu_dtype=gpu_dtype, **kwargs,
        )
        self.pop_size = int(max(pop_size, 3))
        self.sampling = sampling
        self.Z: Optional[np.ndarray] = None
        self.interval: float = 0.0
        self.z_min: Optional[np.ndarray] = None

    def _setup(self, problem, **kwargs):
        Z, n_eff = UniformPoint(self.pop_size, int(problem.n_obj))
        Z = np.asarray(Z, dtype=float)
        order = np.lexsort(Z.T[::-1])          # sortrows
        self.Z = Z[order]
        self.pop_size = int(max(1, n_eff))
        self.interval = float(self.Z[0, -1] - self.Z[1, -1]) if len(self.Z) > 1 else 0.0

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills if infills is not None else Population.empty()
        if len(self.pop) == 0:
            self.opt = self.pop
            return
        self.z_min = np.asarray(self.pop.get("F"), dtype=float).min(axis=0)
        self._set_optimum()

    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)
        cv = self._population_cv(self.pop)
        # emopylab TournamentSelection returns MATLAB-style 1-based indices.
        idx = np.asarray(TournamentSelection(2, self.pop_size, cv), dtype=int) - 1
        X = np.asarray(self.pop.get("X"), dtype=float)
        off = OperatorGA(self.problem, X[idx], rng=self.random_state)
        return Population.new("X", np.asarray(off, dtype=float))

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return
        off_F = np.asarray(infills.get("F"), dtype=float)
        self.z_min = np.minimum(self.z_min, off_F.min(axis=0))
        merged = Population.merge(self.pop, infills)
        self.pop = environmental_selection(
            merged, self.pop_size, self.Z, self.z_min, self.random_state,
        )
        self.Z = adaptive_reference(
            np.asarray(self.pop.get("F"), dtype=float), self.Z, self.pop_size, self.interval,
        )
        self._set_optimum()

    @staticmethod
    def _population_cv(pop: Population) -> np.ndarray:
        G = pop.get("G")
        if G is None:
            return np.zeros(len(pop))
        G = np.asarray(G, dtype=float)
        if G.ndim == 1:
            G = G.reshape(len(pop), -1)
        if G.size == 0:
            return np.zeros(len(pop))
        return np.sum(np.maximum(0.0, G), axis=1)

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)


ALGORITHMS = {"A-NSGA-III": ANSGAIII}
