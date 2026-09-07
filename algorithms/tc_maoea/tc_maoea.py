# -*- coding: utf-8 -*-
# emopylab 2026
"""TC-MaOEA: Tangent-Coupled Many-Objective Evolutionary Algorithm.

A mathematically unified solver for degenerate and irregular many-objective problems.
Couples objective-space manifold SVD tangent bundles with decision-space pullback projectors
and canonical reference-direction environmental selection.
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from operators.utility_functions.NDSort import NDSort
from util.optimum import filter_optimum
from operators.sampling.lhs import LHS
from core.mating import Mating
from operators.crossover.sbx import SBX
from operators.mutation.pm import PolynomialMutation
from operators.selection.tournament import TournamentSelection
from algorithms.moo.sms import cv_and_dom_tournament
from algorithms.nsga3_local.nsga3_local import (
    _environmental_selection,
    _update_zmin,
    _population_objectives,
)
from algorithms.community_utils.moead_family import rng_from_algo
from util.ref_dirs import get_reference_directions

from .tangent_operator import (
    build_pullback_projectors,
    compute_eej_jacobian,
    compute_manifold_tangent_basis,
)

ALGORITHM_FLAGS = {
    "TC-MaOEA": {"multi", "many", "real", "tangent_bundle", "manifold"},
    "TCMaOEA": {"multi", "many", "real", "tangent_bundle", "manifold"},
}


class TCMaOEA(Algorithm):
    """Tangent-Coupled Many-Objective Evolutionary Algorithm (TC-MaOEA).

    Parameters
    ----------
    pop_size : int, default=100
        Population size (fixed at 100).
    ref_dirs : np.ndarray | None, default=None
        Pre-defined reference directions of shape (K, M). If None, generated via get_reference_directions('energy', n_points=100).
    tau : float, default=0.95
        Cumulative energy threshold for intrinsic dimension SVD detection.
    alpha_tangent : float, default=0.25
        Tangent bundle exploration amplification factor.
    """

    ALGO_FLAGS = {"multi", "many", "real", "tangent_bundle", "manifold"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: Optional[np.ndarray] = None,
        tau: float = 0.95,
        alpha_tangent: float = 0.25,
        sampling: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pop_size = int(max(pop_size, 4))
        self.ref_dirs = None if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        self.tau = float(np.clip(tau, 0.50, 0.99))
        self.alpha_tangent = float(alpha_tangent)
        self.sampling = sampling

        # Internal state variables
        self.zmin: Optional[np.ndarray] = None
        self.d_star: int = 1
        self.V_d: np.ndarray = np.empty((0, 0), dtype=float)
        self.sigma: np.ndarray = np.empty(0, dtype=float)
        self.J: np.ndarray = np.empty((0, 0), dtype=float)
        self.P_T: np.ndarray = np.empty((0, 0), dtype=float)
        self.xl: np.ndarray = np.empty(0, dtype=float)
        self.xu: np.ndarray = np.empty(0, dtype=float)

        # Evolutionary mating operators
        self.selection = TournamentSelection(func_comp=cv_and_dom_tournament)
        self.crossover = SBX(prob=1.0, eta=15)
        self.mutation = PolynomialMutation(eta=20)
        self.mating = Mating(
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
        )

    def _setup(self, problem: Any, **kwargs: Any) -> None:
        super()._setup(problem, **kwargs)
        M = int(problem.n_obj)
        D = int(problem.n_var)

        self.xl = np.asarray(problem.xl, dtype=float).copy()
        self.xu = np.asarray(problem.xu, dtype=float).copy()

        # Fixed reference directions (100 points for any M)
        if self.ref_dirs is not None and self.ref_dirs.ndim == 2 and self.ref_dirs.shape[1] == M:
            self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)
            self.pop_size = len(self.ref_dirs)
        else:
            self.ref_dirs = get_reference_directions("energy", n_obj=M, n_points=self.pop_size)

        self.zmin = None
        self.d_star = max(1, M - 1)
        self.P_T = np.eye(D, dtype=float)

    def _initialize_infill(self) -> Population:
        """Generate initial population using Latin Hypercube Sampling."""
        if self.sampling is not None and hasattr(self.sampling, "do"):
            return self.sampling.do(self.problem, self.pop_size)
        return LHS().do(self.problem, self.pop_size)

    def _initialize_advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            self.pop = Population.empty()
            self.opt = self.pop
            return

        self.pop = infills
        self.zmin = _update_zmin(self.zmin, self.pop, int(self.problem.n_obj))
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    def _infill(self) -> Optional[Population]:
        """Generate offspring population using Tangent-Bundle Pullback Variation."""
        if self.pop is None or len(self.pop) == 0:
            return self._initialize_infill()

        # 1. Generate base evolutionary candidates via standard mating
        offspring_raw = self.mating.do(self.problem, self.pop, self.pop_size)
        X_raw = np.asarray(offspring_raw.get("X"), dtype=float)
        X_pop = np.asarray(self.pop.get("X"), dtype=float)

        # 2. Tangent-Coupled Guidance:
        # Project displacement onto tangent bundle P_T to accelerate manifold alignment,
        # while keeping full evolutionary variation on the normal distance subspace
        delta = X_raw - X_pop
        delta_tangent = delta @ self.P_T.T
        delta_normal = delta - delta_tangent

        # Tangent amplification factor along the manifold
        X_off = X_pop + (1.0 + self.alpha_tangent) * delta_tangent + delta_normal
        X_off = np.clip(X_off, self.xl, self.xu)

        return Population.new("X", X_off)

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        """Advance algorithm state, refresh manifold tangent bundle, and select survivors."""
        if infills is None or len(infills) == 0:
            return

        merged = Population.merge(self.pop, infills)
        self.zmin = _update_zmin(self.zmin, merged, int(self.problem.n_obj))

        F_merged = _population_objectives(merged)
        X_merged = np.asarray(merged.get("X"), dtype=float)

        # Refresh tangent bundle using current elite non-dominated front
        front_no, _ = NDSort(F_merged, len(F_merged))
        nd_idx = np.where(np.asarray(front_no).reshape(-1) == 1)[0]
        if len(nd_idx) < 2:
            nd_idx = np.arange(min(len(F_merged), self.pop_size))

        z_nad = np.max(F_merged[nd_idx], axis=0)
        self.d_star, self.V_d, self.sigma = compute_manifold_tangent_basis(
            F_merged[nd_idx], self.zmin, z_nad, tau=self.tau
        )
        self.J = compute_eej_jacobian(X_merged[nd_idx], F_merged[nd_idx])
        self.P_T, _ = build_pullback_projectors(self.J, self.V_d)

        # Environmental selection using canonical NSGA-3 selection on ref_dirs
        selected = _environmental_selection(
            merged,
            self.pop_size,
            np.asarray(self.ref_dirs, dtype=float),
            np.asarray(self.zmin, dtype=float),
            rng_from_algo(self),
        )
        self.pop = selected
        self.opt = filter_optimum(self.pop, least_infeasible=True)


ALGORITHMS = {
    "TC-MaOEA": TCMaOEA,
    "TCMaOEA": TCMaOEA,
}
