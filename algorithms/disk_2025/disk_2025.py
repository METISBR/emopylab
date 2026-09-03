# -*- coding: utf-8 -*-
"""
DISK_2025 (inspired baseline)
=============================

Reference publication:
Z. Zhang, Y. Wang, G. Sun, and T. Pang,
"A Distribution Information-Based Kriging-Assisted Evolutionary Algorithm
for Expensive Many-Objective Optimization Problems,"
IEEE Transactions on Evolutionary Computation, 2025.
DOI: 10.1109/TEVC.2024.3519185

This implementation targets reproducible benchmarking in EmoPyLab:
- Kriging/Gaussian-process objective surrogates
- Candidate pre-screening by surrogate mean + uncertainty
- Distribution pressure through reference-direction niche scarcity

It is intentionally compact and practical, not a line-by-line port.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from util.array_backend import xp as np

from algorithms.moo.nsga3 import ReferenceDirectionSurvival, associate_to_niches
from algorithms.moo.sms import cv_and_dom_tournament
from core.algorithm import Algorithm
from core.duplicate import DefaultDuplicateElimination
from core.mating import Mating
from core.population import Population
from operators.crossover.sbx import SBX
from operators.mutation.pm import PolynomialMutation
from operators.selection.tournament import TournamentSelection
from util.nds.non_dominated_sorting import NonDominatedSorting

try:
    from util.ref_dirs import get_reference_directions as _get_reference_directions
except Exception:  # pragma: no cover
    _get_reference_directions = None
ALGORITHM_FLAGS = {
    "DISK_2025": {"multi", "many"},
}

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    _SKLEARN_OK = True
except Exception:  # pragma: no cover
    _SKLEARN_OK = False


def _force_cpu_backend(algorithm, reason: str) -> None:
    if not bool(getattr(algorithm, "use_gpu", False)):
        return
    algorithm.use_gpu = False
    algorithm.array_backend_effective = "numpy"
    state = dict(getattr(algorithm, "backend_state", {}) or {})
    state["forced_cpu_reason"] = str(reason)
    algorithm.backend_state = state


def _auto_reference_directions(n_obj: int, target_n: int, rng_seed: int = 1) -> np.ndarray:
    """Build reference directions matching the problem objective count."""
    n_obj = int(max(n_obj, 1))
    target_n = int(max(target_n, 2))

    if n_obj == 1:
        return np.ones((1, 1), dtype=float)

    if _get_reference_directions is not None:
        # Prefer methods that can honor the requested count more closely.
        for kwargs in (
            {"method": "energy", "n_points": target_n, "seed": int(rng_seed)},
            {"method": "energy-layer", "n_points": target_n, "seed": int(rng_seed)},
        ):
            try:
                ref = _get_reference_directions(
                    kwargs.pop("method"), n_obj, **kwargs
                )
                ref = np.asarray(ref, dtype=float)
                if ref.ndim == 2 and ref.shape[1] == n_obj and len(ref) > 0:
                    return ref
            except Exception:
                pass

        # Das-Dennis fallback: choose the smallest partition count that reaches target_n.
        best_p = 1
        for p in range(1, 65):
            count = math.comb(p + n_obj - 1, n_obj - 1)
            best_p = p
            if count >= target_n:
                break
        try:
            ref = _get_reference_directions("das-dennis", n_obj, n_partitions=best_p)
            ref = np.asarray(ref, dtype=float)
            if ref.ndim == 2 and ref.shape[1] == n_obj and len(ref) > 0:
                return ref
        except Exception:
            pass

    # Last-resort simplex sampling fallback.
    rng = np.random.default_rng(int(rng_seed))
    ref = rng.random((target_n, n_obj))
    ref_sum = np.clip(np.sum(ref, axis=1, keepdims=True), 1e-12, None)
    return np.asarray(ref / ref_sum, dtype=float)


class DISK_2025(Algorithm):
    """
    Surrogate-assisted many-objective baseline inspired by DISK (2025).
    """
    ALGO_FLAGS = {"multi", "many"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        ref_dirs: np.ndarray,
        pop_size: Optional[int] = None,
        infill_ratio: float = 0.50,
        candidate_multiplier: int = 6,
        min_training_size: int = 60,
        max_training_size: int = 450,
        gp_restarts: int = 0,
        uncertainty_weight: float = 0.20,
        rarity_weight: float = 0.30,
        distance_weight: float = 0.10,
        seed: Optional[int] = None,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            use_gpu=use_gpu,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
            **kwargs,
        )

        _force_cpu_backend(
            self,
            "DISK_2025 surrogate pipeline depends on sklearn GPR CPU stack; GPU backend is pending migration.",
        )

        self.ref_dirs = np.asarray(ref_dirs, dtype=float)
        self.pop_size = int(pop_size) if pop_size is not None else len(self.ref_dirs)

        self.infill_ratio = float(np.clip(infill_ratio, 0.15, 1.0))
        self.candidate_multiplier = int(max(candidate_multiplier, 2))
        self.min_training_size = int(max(min_training_size, 20))
        self.max_training_size = int(max(max_training_size, self.min_training_size))
        self.gp_restarts = int(max(gp_restarts, 0))

        self.uncertainty_weight = float(np.clip(uncertainty_weight, 0.0, 1.0))
        self.rarity_weight = float(np.clip(rarity_weight, 0.0, 1.0))
        self.distance_weight = float(np.clip(distance_weight, 0.0, 1.0))

        self.survival = ReferenceDirectionSurvival(self.ref_dirs)
        self.nds = NonDominatedSorting()
        self.dup_elim = DefaultDuplicateElimination()

        self.selection = TournamentSelection(func_comp=cv_and_dom_tournament)
        self.sbx = SBX(prob=0.9, eta=20, vtype=float)
        self.pm = PolynomialMutation(prob=0.1, eta=20)
        self.mating = Mating(
            self.selection,
            self.sbx,
            self.pm,
            eliminate_duplicates=self.dup_elim,
            n_max_iterations=100,
        )

        self.archive_X: Optional[np.ndarray] = None
        self.archive_F: Optional[np.ndarray] = None
        self.models = []

    def _setup(self, problem, **kwargs):
        if self.ref_dirs.ndim != 2 or self.ref_dirs.shape[1] != problem.n_obj:
            self.ref_dirs = _auto_reference_directions(
                n_obj=problem.n_obj,
                target_n=max(self.pop_size, 4),
                rng_seed=int(getattr(self, "seed", 1) or 1),
            )
            self.survival = ReferenceDirectionSurvival(self.ref_dirs)

        self.pm = PolynomialMutation(prob=min(1.0, 1.0 / max(problem.n_var, 1)), eta=20)
        self.mating = Mating(
            self.selection,
            self.sbx,
            self.pm,
            eliminate_duplicates=self.dup_elim,
            n_max_iterations=100,
        )

    def _initialize_infill(self):
        xl, xu = self.problem.xl, self.problem.xu
        X = xl + self.random_state.random((self.pop_size, self.problem.n_var)) * (xu - xl)
        return Population.new("X", X)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(
            self.problem,
            infills,
            n_survive=self.pop_size,
            random_state=self.random_state,
        )
        self._append_archive(self.pop)
        self._set_optimum()

    def _infill(self):
        n_infill = int(round(self.pop_size * self.infill_ratio))
        n_infill = int(np.clip(n_infill, 2, self.pop_size))

        n_candidates = max(self.pop_size, n_infill * self.candidate_multiplier)
        candidates = self._generate_candidates(n_candidates)

        if len(candidates) == 0:
            return self._random_population(n_infill)

        if len(candidates) < n_infill:
            extra = self._random_population(n_infill - len(candidates))
            candidates = Population.merge(candidates, extra)

        if not self._can_train_surrogate():
            idx = self.random_state.choice(np.arange(len(candidates)), size=n_infill, replace=False)
            return candidates[np.asarray(idx, dtype=int)]

        X_cand = candidates.get("X")
        mu, std = self._surrogate_predict(X_cand)
        score = self._acquisition_score(mu, std)
        pick = np.argsort(score)[:n_infill].astype(int)
        return Population.new("X", X_cand[pick])

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return

        self._append_archive(infills)

        pool = Population.merge(self.pop, infills)
        pool = self.dup_elim.do(pool)
        self.pop = self.survival.do(
            self.problem,
            pool,
            n_survive=self.pop_size,
            random_state=self.random_state,
        )
        self._set_optimum()

    def _set_optimum(self):
        if self.pop is None or len(self.pop) == 0:
            self.opt = self.pop
            return
        nd = self.nds.do(self.pop.get("F"), only_non_dominated_front=True)
        self.opt = self.pop[np.asarray(nd, dtype=int)]

    def _generate_candidates(self, n_candidates: int) -> Population:
        n_sbx = int(round(0.7 * n_candidates))
        n_sbx = int(np.clip(n_sbx, 2, n_candidates))

        off = self.mating.do(
            self.problem,
            self.pop,
            n_sbx,
            algorithm=self,
            random_state=self.random_state,
        )

        if off is None:
            off = Population.new("X", np.empty((0, self.problem.n_var), dtype=float))

        if len(off) > n_sbx:
            off = off[:n_sbx]

        n_rand = max(0, n_candidates - len(off))
        rand = self._random_population(n_rand)

        cand = Population.merge(off, rand) if len(off) > 0 else rand
        cand = self.dup_elim.do(cand)

        if len(cand) == 0:
            return self._random_population(max(2, n_candidates))
        if len(cand) < n_candidates:
            extra = self._random_population(n_candidates - len(cand))
            cand = Population.merge(cand, extra)
        return cand

    def _random_population(self, n: int) -> Population:
        if n <= 0:
            return Population.new("X", np.empty((0, self.problem.n_var), dtype=float))
        xl, xu = self.problem.xl, self.problem.xu
        X = xl + self.random_state.random((n, self.problem.n_var)) * (xu - xl)
        return Population.new("X", X)

    def _append_archive(self, pop: Population):
        X = np.asarray(pop.get("X"), dtype=float)
        F = np.asarray(pop.get("F"), dtype=float)

        if self.archive_X is None:
            self.archive_X = X.copy()
            self.archive_F = F.copy()
            return

        X_all = np.vstack((self.archive_X, X))
        F_all = np.vstack((self.archive_F, F))

        # Decision-space uniqueness with fixed precision for numeric stability.
        X_key = np.round(X_all, decimals=12)
        _, uniq_idx = np.unique(X_key, axis=0, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        self.archive_X = X_all[uniq_idx]
        self.archive_F = F_all[uniq_idx]

    def _can_train_surrogate(self) -> bool:
        if not _SKLEARN_OK:
            return False
        if self.archive_X is None or self.archive_F is None:
            return False
        return len(self.archive_X) >= self.min_training_size

    def _surrogate_predict(self, X_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_train = self.archive_X
        F_train = self.archive_F
        n_train = len(X_train)

        if n_train > self.max_training_size:
            idx = self.random_state.choice(
                np.arange(n_train), size=self.max_training_size, replace=False
            )
            X_train = X_train[idx]
            F_train = F_train[idx]

        xl, xu = self.problem.xl, self.problem.xu
        scale = np.maximum(xu - xl, 1e-12)
        Xs = (X_train - xl) / scale
        Xq = (X_query - xl) / scale

        m = self.problem.n_obj
        mu = np.empty((len(Xq), m), dtype=float)
        std = np.empty((len(Xq), m), dtype=float)

        for j in range(m):
            y = F_train[:, j]
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, nu=2.5)
                + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e-2))
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-10,
                normalize_y=True,
                n_restarts_optimizer=self.gp_restarts,
                random_state=int(self.random_state.integers(1, 2**31 - 1)),
            )
            try:
                model.fit(Xs, y)
                mj, sj = model.predict(Xq, return_std=True)
            except Exception:
                # Robust fallback if GP fitting becomes ill-conditioned.
                y_mean = float(np.mean(y))
                y_std = float(np.std(y))
                mj = np.full(len(Xq), y_mean, dtype=float)
                sj = np.full(len(Xq), max(y_std, 1e-8), dtype=float)

            mu[:, j] = np.asarray(mj, dtype=float)
            std[:, j] = np.asarray(sj, dtype=float)

        std = np.clip(std, 1e-12, None)
        return mu, std

    def _acquisition_score(self, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        F_pop = np.asarray(self.pop.get("F"), dtype=float)
        ideal = np.min(F_pop, axis=0)
        nadir = np.max(F_pop, axis=0)
        span = np.maximum(nadir - ideal, 1e-12)

        pop_niche, _dist, _ = associate_to_niches(F_pop, self.ref_dirs, ideal, nadir)
        niche_count = np.bincount(pop_niche.astype(int), minlength=len(self.ref_dirs))

        cand_niche, cand_dist, _ = associate_to_niches(mu, self.ref_dirs, ideal, nadir)
        rarity = 1.0 / (1.0 + niche_count[cand_niche.astype(int)])

        mu_norm = (mu - ideal) / span
        conv = np.mean(np.clip(mu_norm, -5.0, 5.0), axis=1)

        std_scale = np.maximum(np.std(self.archive_F, axis=0), 1e-12)
        unc = np.mean(std / std_scale, axis=1)
        unc = self._normalize01(unc)

        dist = self._normalize01(cand_dist)
        rarity = self._normalize01(rarity)

        nd_idx = self.nds.do(mu, only_non_dominated_front=True)
        nd_bonus = np.zeros(len(mu), dtype=float)
        nd_bonus[np.asarray(nd_idx, dtype=int)] = 0.08

        score = (
            conv
            + self.distance_weight * dist
            - self.rarity_weight * rarity
            - self.uncertainty_weight * unc
            - nd_bonus
        )
        return np.asarray(score, dtype=float)

    @staticmethod
    def _normalize01(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo <= 1e-14:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo)
