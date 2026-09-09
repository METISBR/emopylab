# -*- coding: utf-8 -*-
# Author: Prof. Thiago Santos & METISBr Research Group, 2026
"""Tangent-Coupled Many-Objective Evolutionary Algorithm (TC-MaOEA).

Implements the differential tangent-bundle pullback architecture
for degenerate and irregular Pareto geometries.
"""
from __future__ import annotations

from typing import Any, List, Optional
import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from operators.utility_functions.UniformPoint import UniformPoint
from util.array_backend import to_numpy
from util.nds.non_dominated_sorting import NonDominatedSorting
from algorithms.community_utils.moead_family import sample_initial

from .tangent_operator import (
    project_simplex,
    certify_empirical_descent,
    solve_kkt_simplex_qp,
    svd_manifold_decomposition,
    compute_eej,
    build_pullback_projectors,
    reflective_clamp,
    apd_environmental_selection,
    population_matrix,
)

ALGORITHM_FLAGS = {
    "TC_MaOEA": {"multi", "many", "real", "constrained"},
    "TCMaOEA": {"multi", "many", "real", "constrained"},
}

__all__ = [
    "TC_MaOEA",
    "TCMaOEA",
    "project_simplex",
    "certify_empirical_descent",
    "solve_kkt_simplex_qp",
    "svd_manifold_decomposition",
    "compute_eej",
    "build_pullback_projectors",
    "reflective_clamp",
    "apd_environmental_selection",
]


class TC_MaOEA(Algorithm):
    """Tangent-Coupled Many-Objective Evolutionary Algorithm (TC-MaOEA).

    Parameters:
        pop_size: Population size (default: 100).
        ref_dirs: Explicit reference directions of shape (K, M). When provided,
                  pop_size matches len(ref_dirs) exactly.
        tau_var: Relative variance floor for SVD dimension estimation (default: 1e-3).
        tau_gap: Spectral eigengap threshold for SVD dimension estimation (default: 50.0).
        reg_scale: Tikhonov regularization scaling for EEJ Jacobian (default: 1e-6).
        gamma: Regularization for damped right inverse (default: 1e-8).
        alpha: APD penalty exponent transitioning diversity to convergence (default: 2.0).
        scale_normal_step: Whether to scale normal KKT step by mean DE displacement (default: True).
        early_normal_cap: Whether to cap normal weight eta_N at 0.3 during the first
                          30% of the evaluation budget (default: False).
    """

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: Optional[np.ndarray] = None,
        tau_var: float = 1e-3,
        tau_gap: float = 50.0,
        reg_scale: float = 1e-6,
        gamma: float = 1e-8,
        alpha: float = 2.0,
        scale_normal_step: bool = True,
        early_normal_cap: bool = False,
        sampling=None,
        seed: Optional[int] = None,
        use_gpu: bool = False,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            seed=seed,
            use_gpu=use_gpu,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
            **kwargs,
        )
        self.pop_size = int(max(pop_size, 4))
        self.ref_dirs = ref_dirs
        self.tau_var = float(tau_var)
        self.tau_gap = float(tau_gap)
        self.reg_scale = float(reg_scale)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.scale_normal_step = bool(scale_normal_step)
        self.early_normal_cap = bool(early_normal_cap)
        self.sampling = sampling

        # Internal state
        self.W: Optional[np.ndarray] = None
        self.W_adapt: Optional[np.ndarray] = None
        self.z_min: Optional[np.ndarray] = None
        self.z_max: Optional[np.ndarray] = None
        self.d_star: int = 2
        self.sigmas: Optional[np.ndarray] = None
        self.P_T: Optional[np.ndarray] = None
        self.P_N: Optional[np.ndarray] = None
        self.d_kkt: Optional[np.ndarray] = None
        self.eta_T: float = 1.0
        self.eta_N: float = 0.0

    def _setup(self, problem: Any, **kwargs: Any) -> None:
        M = int(problem.n_obj)
        if self.ref_dirs is not None:
            self.W = np.asarray(to_numpy(self.ref_dirs), dtype=float)
            self.pop_size = len(self.W)
        else:
            w_init, n_eff = UniformPoint(self.pop_size, M)
            w_arr = np.asarray(w_init, dtype=float)
            if len(w_arr) == self.pop_size:
                self.W = w_arr
            else:
                try:
                    from util.ref_dirs import get_reference_directions
                    self.W = get_reference_directions("energy", n_obj=M, n_points=self.pop_size)
                except Exception:
                    self.W = w_arr
                    self.pop_size = len(self.W)
        self.W_adapt = np.copy(self.W)

    def _initialize_infill(self) -> Optional[Population]:
        return sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)

    def _initialize_advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        self.pop = infills if infills is not None else Population.empty()
        if len(self.pop) == 0:
            self.opt = self.pop
            return

        F = population_matrix(self.pop, "F")
        self.z_min = np.min(F, axis=0)
        self.z_max = np.max(F, axis=0)
        self._update_tangent_operators()
        self._set_optimum()

    def _set_fallback_operators(self, D: int) -> None:
        self.W_adapt = np.copy(self.W) if self.W is not None else None
        self.P_T = np.eye(D)
        self.P_N = np.zeros((D, D))
        self.d_kkt = np.zeros(D)
        self.eta_T = 1.0
        self.eta_N = 0.0

    def _update_tangent_operators(self) -> None:
        """Executes SVD manifold decomposition, reference vector adaptation,

        EEJ regression, pullback projector construction, and KKT descent resolution.
        """
        if self.pop is None or len(self.pop) < 2:
            return

        X = population_matrix(self.pop, "X")
        F = population_matrix(self.pop, "F")
        M = int(self.problem.n_obj)
        D = int(X.shape[1])

        self.z_min = np.minimum(self.z_min, np.min(F, axis=0))
        self.z_max = np.maximum(self.z_max, np.max(F, axis=0))

        # True non-dominated front count before pooling
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        n_nd = len(fronts[0]) if len(fronts) > 0 else 0

        # Elite sample pooling for statistical estimation (requires at least max(4, M) points)
        elite_idx = np.array(fronts[0], dtype=int) if len(fronts) > 0 else np.array([], dtype=int)
        if len(elite_idx) < max(4, M):
            pooled: List[int] = []
            for fr in fronts:
                pooled.extend(fr)
                if len(pooled) >= max(4, M):
                    break
            elite_idx = np.array(pooled, dtype=int)

        if len(elite_idx) < 2:
            self._set_fallback_operators(D)
            return

        X_elite = X[elite_idx]
        F_elite = F[elite_idx]

        span = np.maximum(self.z_max - self.z_min, 1e-12)
        F_norm = (F_elite - self.z_min[None, :]) / span[None, :]

        # 1. Spectral Manifold Decomposition
        d_star, V_dstar, sigmas = svd_manifold_decomposition(
            F_norm, tau_var=self.tau_var, tau_gap=self.tau_gap
        )
        self.d_star = d_star
        self.sigmas = sigmas

        sigma_1 = float(sigmas[0]) if len(sigmas) > 0 else 0.0
        sigma_d = float(sigmas[d_star - 1]) if (d_star >= 1 and len(sigmas) >= d_star) else 0.0
        finite_spectrum = bool(np.all(np.isfinite(sigmas))) and (sigma_1 > 0)
        sigma_ratio = (sigma_d / sigma_1) if (finite_spectrum and sigma_1 > 0) else 0.0

        # Two-sided gate: N_ND >= 3*M, finite spectrum, d_star >= 1, spectral health ratio > 1e-2
        gate_ok = (n_nd >= 3 * M) and finite_spectrum and (d_star >= 1) and (sigma_ratio > 1e-2)

        if not gate_ok:
            self._set_fallback_operators(D)
            return

        # 2. Reference Vector Manifold Projection with row fallback
        W_proj = self.W @ V_dstar @ V_dstar.T
        norm_proj = np.linalg.norm(W_proj, axis=1, keepdims=True)
        valid_norm = norm_proj[:, 0] > 1e-12
        W_adapt = np.copy(self.W)
        W_adapt[valid_norm] = W_proj[valid_norm] / norm_proj[valid_norm]
        self.W_adapt = W_adapt

        # 3. Ensemble Evolutionary Jacobian (EEJ)
        J = compute_eej(X_elite, F_norm, reg_scale=self.reg_scale)

        # 4. Pullback Projectors in Decision Space
        P_T, P_N = build_pullback_projectors(J, V_dstar, gamma=self.gamma)
        self.P_T = P_T
        self.P_N = P_N

        # 5. Dual Simplex KKT Pareto Descent Vector
        _, d_kkt = solve_kkt_simplex_qp(J)
        self.d_kkt = d_kkt

        # 6. Dynamic Spectral Balancing
        eta_T = float(np.clip(np.sqrt(sigma_ratio), 0.0, 1.0))
        eta_N = 1.0 - eta_T

        if self.early_normal_cap:
            curr_eval = float(self.n_evals)
            max_eval = float(getattr(self.termination, "n_max_evals", getattr(self.termination, "n_max_eval", 30000)))
            t_ratio = float(np.clip(curr_eval / max(max_eval, 1.0), 0.0, 1.0))
            if t_ratio < 0.3:
                eta_N = min(eta_N, 0.3)
                eta_T = 1.0 - eta_N

        self.eta_T = eta_T
        self.eta_N = eta_N

    def _compute_delta_x(self, delta_T: np.ndarray, delta_N: np.ndarray, N: int, D: int) -> np.ndarray:
        return self.eta_T * delta_T + self.eta_N * delta_N[None, :]

    def _infill(self) -> Optional[Population]:
        if self.pop is None or len(self.pop) == 0:
            return sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)

        X = population_matrix(self.pop, "X")
        N, D = X.shape
        xl = np.asarray(to_numpy(self.problem.xl), dtype=float)
        xu = np.asarray(to_numpy(self.problem.xu), dtype=float)

        P_T = self.P_T if self.P_T is not None else np.eye(D)
        P_N = self.P_N if self.P_N is not None else np.zeros((D, D))
        d_kkt = self.d_kkt if self.d_kkt is not None else np.zeros(D)

        arange_N = np.arange(N)
        if N >= 3:
            if hasattr(self.random_state, "integers"):
                r1 = self.random_state.integers(0, N - 1, size=N)
            else:
                r1 = self.random_state.randint(0, N - 1, size=N)
            r1[r1 >= arange_N] += 1

            if hasattr(self.random_state, "integers"):
                r2 = self.random_state.integers(0, N - 2, size=N)
            else:
                r2 = self.random_state.randint(0, N - 2, size=N)
            min_r = np.minimum(arange_N, r1)
            max_r = np.maximum(arange_N, r1)
            r2[r2 >= min_r] += 1
            r2[r2 >= max_r] += 1
        else:
            r1 = (arange_N + 1) % N
            r2 = (arange_N + 2) % N

        delta_de = X[r1] - X[r2]
        delta_T = delta_de @ P_T.T

        pn_dkkt = P_N @ d_kkt
        if self.scale_normal_step:
            norm_pn = float(np.linalg.norm(pn_dkkt))
            mean_de = float(np.mean(np.linalg.norm(delta_de, axis=1)))
            scale = mean_de / (norm_pn + 1e-8)
            delta_N = scale * pn_dkkt
        else:
            delta_N = pn_dkkt

        delta_x = self._compute_delta_x(delta_T, delta_N, N, D)
        offspring_X = reflective_clamp(X + delta_x, xl, xu)
        return Population.new("X", offspring_X)

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> Any:
        if infills is None or len(infills) == 0:
            return

        off_F = population_matrix(infills, "F")
        self.z_min = np.minimum(self.z_min, np.min(off_F, axis=0))
        self.z_max = np.maximum(self.z_max, np.max(off_F, axis=0))

        merged = Population.merge(self.pop, infills)

        curr_eval = float(self.n_evals)
        max_eval = float(getattr(self.termination, "n_max_evals", getattr(self.termination, "n_max_eval", 30000)))
        t_ratio = float(np.clip(curr_eval / max(max_eval, 1.0), 0.0, 1.0))

        self.pop = apd_environmental_selection(
            pool=merged,
            W_adapt=self.W_adapt if self.W_adapt is not None else self.W,
            z_min=self.z_min,
            z_max=self.z_max,
            n_survive=self.pop_size,
            t_ratio=t_ratio,
            alpha=self.alpha,
        )

        self._update_tangent_operators()
        self._set_optimum()

    def _set_optimum(self) -> None:
        if self.pop is None or len(self.pop) == 0:
            return
        F = population_matrix(self.pop, "F")
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        if len(fronts) > 0 and len(fronts[0]) > 0:
            self.opt = self.pop[np.array(fronts[0], dtype=int)]
        else:
            self.opt = self.pop


TCMaOEA = TC_MaOEA
