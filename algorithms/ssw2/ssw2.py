"""SSW2: Population-Based Stochastic Steepest Weights with Ensemble Evolutionary Jacobian.

Evolutionary vector optimization framework combining:
1. Ensemble Evolutionary Jacobian (gradient-free O(0) additional function evaluation cost).
2. Adaptive step-size (sigma) and noise intensity (epsilon) via 3 selectable strategies:
   - "cma_csa": CMA-ES Cumulative Step-Size Adaptation with path integration.
   - "success_rule": 1/5th Success Rule with exponential smoothing.
   - "spectral_cosine": Barzilai-Borwein Spectral Step with Cosine Annealing.
3. NSGA-III structured reference directions for niching and manifold diversity.
4. Non-dominated Pareto archive tracking with crowding distance truncation.

Authors: Prof. Thiago Santos & METISBr Research Group (UFOP / 2026).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from core.algorithm import Algorithm
from core.population import Population
from operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from algorithms.nsga3.nsga3 import ReferenceDirectionSurvival
from util.display.multi import MultiObjectiveOutput
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.optimum import filter_optimum
from util.ref_dirs import get_reference_directions


def _project_to_simplex_batch(v: np.ndarray) -> np.ndarray:
    """Vectorized Euclidean projection of rows onto the probability simplex."""
    values = np.asarray(v, dtype=float)
    if values.shape[1] == 1:
        return np.ones_like(values)
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=float)
    positive = ordered - cumulative / indices[np.newaxis, :] > 0.0
    rho = np.maximum(np.sum(positive, axis=1) - 1, 0).astype(int)
    rows = np.arange(values.shape[0])
    theta = cumulative[rows, rho] / (rho + 1.0)
    projected = np.maximum(values - theta[:, np.newaxis], 0.0)
    total = np.sum(projected, axis=1, keepdims=True)
    return projected / np.maximum(total, 1e-16)


def _compute_q_batch(jacobians: np.ndarray, max_iter: int = 150, tol: float = 1e-8) -> np.ndarray:
    """Compute common descent vectors q(x) = J(x)^T @ alpha* for a population batch."""
    batch = np.asarray(jacobians, dtype=float)
    n_points, n_obj, _ = batch.shape
    if n_obj == 1:
        return batch[:, 0, :]

    gram = batch @ np.swapaxes(batch, 1, 2)  # (N, m, m)
    alpha = np.full((n_points, n_obj), 1.0 / n_obj, dtype=float)
    lipschitz = np.max(np.sum(np.abs(gram), axis=2), axis=1)
    step = 1.0 / np.maximum(lipschitz, 1e-12)

    for _ in range(max_iter):
        previous = alpha
        gradient = np.einsum("bij,bj->bi", gram, alpha)
        alpha = _project_to_simplex_batch(alpha - step[:, np.newaxis] * gradient)
        if float(np.max(np.linalg.norm(alpha - previous, axis=1))) <= tol:
            break
    return np.einsum("bmn,bm->bn", batch, alpha)


class SSW2(Algorithm):
    """SSW2: Adaptive Population-Based Stochastic Steepest Weights Optimizer."""

    ALGO_FLAGS = {"multi", "many", "evolutionary", "stochastic_differential"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        pop_size: int = 100,
        step_strategy: str = "spectral_cosine",  # "spectral_cosine", "cma_csa", "success_rule"
        step_size_init: float = 0.1,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        epsilon_init: float = 0.02,
        epsilon_min: float = 1e-4,
        epsilon_max: float = 0.25,
        ref_dirs: Optional[np.ndarray] = None,
        k_neighbors: Optional[int] = None,
        regularization: float = 1e-5,
        use_momentum: bool = True,
        momentum_beta: float = 0.85,
        tangential_diffusion: bool = False,
        output=MultiObjectiveOutput(),
        array_backend: str = "numpy",
        gpu_dtype: str = "float32",
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(
            output=output,
            use_gpu=use_gpu,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
            **kwargs,
        )
        self.pop_size = int(pop_size)
        self.step_strategy = str(step_strategy).strip().lower()
        self.step_size = float(step_size_init)
        self.sigma = float(step_size_init)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.epsilon = float(epsilon_init)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_max = float(epsilon_max)
        self.ref_dirs = ref_dirs
        self.k_neighbors = k_neighbors
        self.regularization = float(regularization)
        self.use_momentum = bool(use_momentum)
        self.momentum_beta = float(momentum_beta)
        self.tangential_diffusion = bool(tangential_diffusion)
        self.archive_size = int(pop_size)
        self.V: Optional[np.ndarray] = None
        # CMA-ES Path Integration (CSA) state
        self.p_sigma: Optional[np.ndarray] = None
        self.c_sigma: float = 0.2
        self.d_sigma: float = 1.5
        self.chi_n: float = 1.0

        # Success Rule state
        self.p_succ: float = 0.2
        self.c_p: float = 0.1
        self.p_target: float = 0.2

        # Population archive and ideal point tracking
        self.ideal_point: Optional[np.ndarray] = None
        self.nadir_point: Optional[np.ndarray] = None
        self.ssw_archive: Optional[Population] = None
        self.history_metrics: List[Dict[str, float]] = []

    def _setup(self, problem, **kwargs):
        n_var = int(problem.n_var)
        n_obj = int(problem.n_obj)

        # Dimension expectation for standard normal vector norm
        self.chi_n = math.sqrt(n_var) * (1.0 - 1.0 / (4.0 * n_var) + 1.0 / (21.0 * n_var**2))
        self.c_sigma = (self.c_sigma or 4.0 / (n_var + 4.0))
        self.d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((n_var - 1.0) / (n_obj + 1.0)) - 1.0) + self.c_sigma
        self.p_sigma = np.zeros(n_var, dtype=float)

        # Resolve NSGA-III reference directions
        if self.ref_dirs is None:
            if n_obj <= 2:
                from operators.utility_functions.UniformPoint import UniformPoint
                self.ref_dirs, n_eff = UniformPoint(self.pop_size, n_obj)
                self.pop_size = int(n_eff)
            elif n_obj <= 3:
                self.ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            elif n_obj <= 5:
                self.ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)
            else:
                self.ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=3)

        self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)
        self.survival = ReferenceDirectionSurvival(ref_dirs=self.ref_dirs)
        self.nds = NonDominatedSorting()
        self.k_neighbors = self.k_neighbors or max(n_obj + 2, min(self.pop_size - 1, 2 * n_obj + int(math.sqrt(n_var))))
        self.archive_size = self.pop_size
        self.V = np.zeros((self.pop_size, n_var), dtype=float)

    def _initialize_infill(self):
        """Sample initial population uniformly in decision space."""
        lower = np.asarray(self.problem.xl, dtype=float)
        upper = np.asarray(self.problem.xu, dtype=float)
        random = self.random_state.random((self.pop_size, self.problem.n_var))
        return Population.new("X", lower + random * (upper - lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._update_reference_bounds(infills.get("F"))
        self.ssw_archive = filter_optimum(infills, least_infeasible=True)
        self.opt = self.ssw_archive

    def _update_reference_bounds(self, F: np.ndarray) -> None:
        valid = F[np.all(np.isfinite(F), axis=1)]
        if len(valid) == 0:
            return
        if self.ideal_point is None:
            self.ideal_point = np.min(valid, axis=0)
            self.nadir_point = np.max(valid, axis=0)
        else:
            self.ideal_point = np.minimum(self.ideal_point, np.min(valid, axis=0))
            self.nadir_point = np.maximum(self.nadir_point, np.max(valid, axis=0))

    def _compute_ensemble_jacobians(self, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Estimate Jacobian matrices J_i for every individual with ZERO extra function evaluations."""
        N, n = X.shape
        m = F.shape[1]
        jacobians = np.zeros((N, m, n), dtype=float)

        # Compute pairwise distance matrix in decision space
        dist_matrix = cdist(X, X)
        np.fill_diagonal(dist_matrix, np.inf)

        reg_eye = self.regularization * np.eye(n)

        for i in range(N):
            k = min(self.k_neighbors, N - 1)
            neighbor_indices = np.argpartition(dist_matrix[i], k)[:k]

            dX = X[neighbor_indices] - X[i]  # (k, n)
            dF = F[neighbor_indices] - F[i]  # (k, m)

            # Regularized Ridge Least Squares: dX @ J_i^T = dF  ==> J_i^T = (dX^T dX + reg*I)^-1 @ dX^T dF
            H = dX.T @ dX + reg_eye
            try:
                JT = np.linalg.solve(H, dX.T @ dF)
            except np.linalg.LinAlgError:
                JT = np.linalg.pinv(H) @ (dX.T @ dF)

            jacobians[i] = JT.T

        return jacobians

    def _adapt_parameters_cma_csa(self, Q: np.ndarray) -> None:
        """Strategy 1: CMA-ES Cumulative Path Length Adaptation (CSA)."""
        # Average descent direction across the non-dominated elite
        ranks = self.nds.do(self.pop.get("F"))
        elite_indices = ranks[0] if len(ranks) > 0 else np.arange(len(Q))
        mean_q = np.mean(Q[elite_indices], axis=0)
        norm_q = np.linalg.norm(mean_q)
        n_var = len(mean_q)
        normalized_q = math.sqrt(n_var) * (mean_q / max(norm_q, 1e-12))

        # Exponential moving average of descent path
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + math.sqrt(self.c_sigma * (2.0 - self.c_sigma)) * normalized_q
        path_norm = float(np.linalg.norm(self.p_sigma))

        # Adjust step size: expand if aligned, shrink if oscillating
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (path_norm / self.chi_n - 1.0))
        self.sigma = float(np.clip(self.sigma, self.sigma_min, self.sigma_max))

        # Inverted noise adaptation: high drift when aligned, high Brownian diffusion when stationary
        self.epsilon = self.epsilon_max * math.exp(-2.0 * (path_norm / self.chi_n))
        self.epsilon = float(np.clip(self.epsilon, self.epsilon_min, self.epsilon_max))

    def _adapt_parameters_success_rule(self, n_success: int, total_candidates: int) -> None:
        """Strategy 2: 1/5th Success Rule with exponential smoothing."""
        success_rate = n_success / max(1, total_candidates)
        self.p_succ = (1.0 - self.c_p) * self.p_succ + self.c_p * success_rate

        factor = math.exp((1.0 / self.d_sigma) * (self.p_succ - self.p_target) / (1.0 - self.p_target))
        self.sigma = float(np.clip(self.sigma * factor, self.sigma_min, self.sigma_max))

        if self.p_succ < 0.05:  # Stagnation detected
            self.epsilon = float(np.clip(self.epsilon * 1.5, self.epsilon_min, self.epsilon_max))
        else:
            self.epsilon = float(np.clip(self.epsilon * 0.95, self.epsilon_min, self.epsilon_max))

    def _adapt_parameters_spectral_cosine(self, t: int, t_max: int) -> None:
        """Strategy 3: Cosine Annealing schedule."""
        ratio = min(1.0, max(0.0, t / max(1, t_max)))
        self.sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1.0 + math.cos(math.pi * ratio))
        self.epsilon = self.epsilon_min + 0.5 * (self.epsilon_max - self.epsilon_min) * (1.0 + math.cos(math.pi * ratio))

    def _infill(self):
        X = np.asarray(self.pop.get("X"), dtype=float)
        F = np.asarray(self.pop.get("F"), dtype=float)
        N, n = X.shape

        # 1. Compute Ensemble Evolutionary Jacobians (0 function evaluations!)
        jacobians = self._compute_ensemble_jacobians(X, F)

        # 2. Vectorized QP over simplex for common descent directions Q(X)
        Q = _compute_q_batch(jacobians)

        # 3. Adapt step size and diffusion intensity
        t_cur = getattr(self, "n_gen", 1)
        t_max = max(100, getattr(self.termination, "n_max_gen", 300))

        if self.step_strategy == "cma_csa":
            self._adapt_parameters_cma_csa(Q)
        elif self.step_strategy == "spectral_cosine":
            self._adapt_parameters_spectral_cosine(t_cur, t_max)

        # 4. Underdamped Langevin Momentum (Second-order continuous dynamics)
        if self.use_momentum:
            if self.V is None or self.V.shape != (N, n):
                self.V = -Q.copy()
            else:
                self.V = self.momentum_beta * self.V - (1.0 - self.momentum_beta) * Q
            drift = self.V
        else:
            drift = -Q

        # 5. Riemannian Tangential Diffusion on ker(J) (Manifold-preserving Brownian noise)
        noise = self.random_state.standard_normal((N, n))
        if self.tangential_diffusion and F.shape[1] > 0:
            m = F.shape[1]
            reg_eye = 1e-7 * np.eye(m)
            Gram = jacobians @ np.swapaxes(jacobians, 1, 2) + reg_eye
            J_xi = np.einsum("bmn,bn->bm", jacobians, noise)
            try:
                w = np.linalg.solve(Gram, J_xi[:, :, None])[:, :, 0]
                xi_normal = np.einsum("bmn,bm->bn", jacobians, w)
                eff_noise = noise - xi_normal
            except Exception:
                eff_noise = noise
        else:
            eff_noise = noise

        # 6. Euler-Maruyama SDE Step with Combined Momentum & Tangential Diffusion
        next_X = X + self.sigma * drift + self.epsilon * math.sqrt(max(1e-12, self.sigma)) * eff_noise

        # 7. Boundary handling: project onto box bounds
        if self.problem.has_bounds():
            next_X = np.clip(next_X, self.problem.xl, self.problem.xu)

        return Population.new("X", next_X)

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            return

        # Combine parent and offspring populations (2N individuals)
        combined = Population.merge(self.pop, infills)
        self._update_reference_bounds(combined.get("F"))

        # Update success rate for Strategy 2
        ranks = self.nds.do(combined.get("F"))
        n_offspring_elite = sum(1 for idx in ranks[0] if idx >= len(self.pop))
        if self.step_strategy == "success_rule":
            self._adapt_parameters_success_rule(n_offspring_elite, len(infills))

        # NSGA-III Reference-Direction Niching Survival Selection (2N -> N)
        self.pop = self.survival.do(self.problem, combined, n_survive=self.pop_size)

        # Accumulate global non-dominated Pareto archive
        archive_candidates = (
            self.pop
            if self.ssw_archive is None
            else Population.merge(self.ssw_archive, self.pop)
        )
        non_dom = filter_optimum(archive_candidates, least_infeasible=True)
        self.ssw_archive = self._truncate_archive(non_dom)
        self.opt = self.ssw_archive

    def _truncate_archive(self, archive: Population) -> Population:
        if archive is None or len(archive) == 0:
            return archive
        objectives = np.asarray(archive.get("F"), dtype=float)
        _, unique_indices = np.unique(objectives, axis=0, return_index=True)
        archive = archive[np.sort(unique_indices)]
        if len(archive) <= self.archive_size:
            return archive
        crowding = calc_crowding_distance(np.asarray(archive.get("F"), dtype=float))
        order = np.argsort(-crowding, kind="mergesort")
        return archive[order[: self.archive_size]]

    def _set_optimum(self):
        self.opt = (
            self.ssw_archive
            if self.ssw_archive is not None and len(self.ssw_archive) > 0
            else filter_optimum(self.pop, least_infeasible=True)
        )
