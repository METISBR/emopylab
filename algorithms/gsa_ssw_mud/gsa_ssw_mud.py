"""GSA-SSW-MUD: Population-Based Stochastic Steepest Weights Optimizer with GSA and MUD.

A native evolutionary vector-optimization proposal built on four coupled mechanisms:

1. Population-Based Jacobian Estimation (GSA).
   Jacobian matrices J_i for every individual are estimated from
   nearest-neighbor least-squares differences over the population, at
   ZERO extra function evaluations (fully gradient-free).

2. Underdamped-Langevin momentum.
   A second-order momentum term smooths the common descent direction,
   reducing oscillation and accelerating exploration of the Pareto
   manifold, embedded directly in the Euler-Maruyama update.

3. Adaptive step-size (sigma) and diffusion intensity (epsilon) via
   pure quadratic cosine annealing.

4. Mixture Uniform Design (MUD) reference directions for niching and
   manifold diversity, preserving exact population cardinality across
   arbitrary objective counts M.

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
from operators.sampling.lhs import LatinHypercubeSampling
from operators.utility_functions.OperatorGA import OperatorGA
from algorithms.nsga3.nsga3 import ReferenceDirectionSurvival
from util.display.multi import MultiObjectiveOutput
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.optimum import filter_optimum

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


class GSASSWMUD(Algorithm):
    """GSA-SSW-MUD: Population-Based Stochastic Steepest Weights Optimizer."""

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
        cooling_power: int = 2,
        momentum_tangential: bool = False,
        nesterov_lookahead: bool = False,
        finishing_fraction: float = 0.0,
        finishing_decay: bool = False,
        tangential_diffusion: bool = False,
        use_fisher_preconditioner: bool = False,
        use_entropic_niching: bool = False,
        entropic_gamma_max: float = 0.3,
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
        self.cooling_power = int(cooling_power)
        self.momentum_tangential = bool(momentum_tangential)
        self.nesterov_lookahead = bool(nesterov_lookahead)
        self.finishing_fraction = float(finishing_fraction)
        self.finishing_decay = bool(finishing_decay)
        self.tangential_diffusion = bool(tangential_diffusion)
        self.use_fisher_preconditioner = bool(use_fisher_preconditioner)
        self.use_entropic_niching = bool(use_entropic_niching)
        self.entropic_gamma_max = float(entropic_gamma_max)
        self.archive_size = int(pop_size)
        self.V: Optional[np.ndarray] = None
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

        # Reference directions: Mixture Uniform Design (MUD) exclusively,
        # for every objective count. No fallback generator; any failure
        # in the MUD lattice propagates loudly (fail-fast).
        if self.ref_dirs is None:
            from operators.utility_functions._common import uniform_point
            self.ref_dirs, _ = uniform_point(self.pop_size, n_obj, method="mud")

        self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)
        # Preserve exact requested pop_size without combinatorial override
        self.survival = ReferenceDirectionSurvival(ref_dirs=self.ref_dirs)
        self.nds = NonDominatedSorting()
        self.k_neighbors = self.k_neighbors or max(n_obj + 2, min(self.pop_size - 1, 2 * n_obj + int(math.sqrt(n_var))))
        self.archive_size = self.pop_size
        self.V = np.zeros((self.pop_size, n_var), dtype=float)

    def _initialize_infill(self):
        """Sample initial population via Latin Hypercube Sampling in decision space."""
        lhs = LatinHypercubeSampling()
        return lhs.do(self.problem, self.pop_size, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        init_V = np.zeros((len(infills), self.problem.n_var), dtype=float)
        self.pop.set("V", init_V)
        self.V = init_V
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
        """Estimate Jacobian matrices J_i for every individual with ZERO extra function evaluations (GSA)."""
        N, n = X.shape
        m = F.shape[1]


        # Compute pairwise distance matrix in decision space
        dist_matrix = cdist(X, X)
        np.fill_diagonal(dist_matrix, np.inf)

        reg_eye = self.regularization * np.eye(n)

        # Generic scale invariance across objectives (dimensionless relative rates)
        f_range = np.max(F, axis=0) - np.min(F, axis=0)
        f_scale = np.maximum(f_range, 1.0)
        norm_F = F / f_scale[np.newaxis, :]

        k = min(self.k_neighbors, N - 1)
        neighbor_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
        dX = X[neighbor_indices] - X[:, np.newaxis, :]  # (N, k, n)
        dF = norm_F[neighbor_indices] - norm_F[:, np.newaxis, :]  # (N, k, m)

        # Batched Regularized Ridge Least Squares
        H = np.matmul(dX.swapaxes(1, 2), dX) + reg_eye[np.newaxis, :, :]  # (N, n, n)
        RHS = np.matmul(dX.swapaxes(1, 2), dF)  # (N, n, m)
        try:
            JT = np.linalg.solve(H, RHS)  # (N, n, m)
        except np.linalg.LinAlgError:
            JT = np.linalg.pinv(H) @ RHS

        return JT.swapaxes(1, 2)  # (N, m, n)

    def _adapt_parameters_cma_csa(self, Q: np.ndarray) -> None:
        """Strategy 1: CMA-ES Cumulative Path Length Adaptation (CSA)."""
        ranks = self.nds.do(self.pop.get("F"))
        elite_indices = ranks[0] if len(ranks) > 0 else np.arange(len(Q))
        mean_q = np.mean(Q[elite_indices], axis=0)
        norm_q = np.linalg.norm(mean_q)
        n_var = len(mean_q)
        normalized_q = math.sqrt(n_var) * (mean_q / max(norm_q, 1e-12))

        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + math.sqrt(self.c_sigma * (2.0 - self.c_sigma)) * normalized_q
        path_norm = float(np.linalg.norm(self.p_sigma))

        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (path_norm / self.chi_n - 1.0))
        self.sigma = float(np.clip(self.sigma, self.sigma_min, self.sigma_max))

        self.epsilon = self.epsilon_max * math.exp(-2.0 * (path_norm / self.chi_n))
        self.epsilon = float(np.clip(self.epsilon, self.epsilon_min, self.epsilon_max))

    def _adapt_parameters_success_rule(self, n_success: int, total_candidates: int) -> None:
        """Strategy 2: 1/5th Success Rule with exponential smoothing."""
        success_rate = n_success / max(1, total_candidates)
        self.p_succ = (1.0 - self.c_p) * self.p_succ + self.c_p * success_rate

        factor = math.exp((1.0 / self.d_sigma) * (self.p_succ - self.p_target) / (1.0 - self.p_target))
        self.sigma = float(np.clip(self.sigma * factor, self.sigma_min, self.sigma_max))

        if self.p_succ < 0.05:
            self.epsilon = float(np.clip(self.epsilon * 1.5, self.epsilon_min, self.epsilon_max))
        else:
            self.epsilon = float(np.clip(self.epsilon * 0.95, self.epsilon_min, self.epsilon_max))

    def _adapt_parameters_spectral_cosine(self, t: int, t_max: int) -> None:
        """Dynamic step size sigma(t) and diffusion intensity epsilon(t) via quartic cosine annealing."""
        ratio = min(1.0, max(0.0, t / max(1, t_max)))
        base = 0.5 * (1.0 + math.cos(math.pi * ratio))
        cooling = base ** (2 * self.cooling_power)
        self.sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * cooling
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * cooling
    def _infill(self):
        X = np.asarray(self.pop.get("X"), dtype=float)
        F = np.asarray(self.pop.get("F"), dtype=float)
        N, n = X.shape
        box = (self.problem.xu - self.problem.xl)

        # Strict Budget Control (0% overrun)
        n_offspring = int(self.pop_size)
        if hasattr(self, "evaluator") and hasattr(self, "termination"):
            n_eval = getattr(self.evaluator, "n_eval", 0)
            n_max_evals = getattr(self.termination, "n_max_evals", None)
            if n_max_evals is not None and n_max_evals > 0:
                rem = max(0, n_max_evals - n_eval)
                if 0 < rem < n_offspring:
                    n_offspring = rem

        # 1. Compute Population-Based Jacobians (GSA, 0 oracle evaluations)
        jacobians = self._compute_ensemble_jacobians(X, F)

        # 2. Vectorized QP over simplex for common descent directions Q(X)
        Q = _compute_q_batch(jacobians)

        if self.use_fisher_preconditioner:
            reg_n = 1e-4 * np.eye(n)
            G_fisher = np.swapaxes(jacobians, 1, 2) @ jacobians + reg_n
            try:
                Q_prec = np.linalg.solve(G_fisher, Q[:, :, None])[:, :, 0]
            except Exception:
                Q_prec = Q.copy()
            q_norms = np.linalg.norm(Q_prec, axis=1, keepdims=True)
            Q_bounded = Q_prec / np.maximum(q_norms, 1.0)
        else:
            q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
            Q_bounded = Q / np.maximum(q_norms, 1.0)

        t_cur = getattr(self, "n_gen", 1)
        t_max = max(100, getattr(self.termination, "n_max_gen", 300))

        if self.use_entropic_niching and self.ref_dirs is not None and len(self.ref_dirs) > 0:
            f_min = np.min(F, axis=0)
            f_max = np.max(F, axis=0)
            f_norm = (F - f_min) / np.maximum(f_max - f_min, 1e-6)
            f_norm = f_norm / np.maximum(np.sum(f_norm, axis=1, keepdims=True), 1e-6)

            dist_to_refs = cdist(f_norm, self.ref_dirs)
            closest_ref_idx = np.argmin(dist_to_refs, axis=1)
            target_refs = self.ref_dirs[closest_ref_idx]

            dF_niching = target_refs - f_norm
            F_niching = np.einsum("bmn,bm->bn", jacobians, dF_niching)
            f_nich_norms = np.linalg.norm(F_niching, axis=1, keepdims=True)
            F_nich_normed = F_niching / np.maximum(f_nich_norms, 1.0)

            gamma = min(self.entropic_gamma_max, self.entropic_gamma_max * (t_cur / t_max))
            Q_bounded = (1.0 - gamma) * Q_bounded - gamma * F_nich_normed

        # 3. Adapt step size and diffusion intensity
        if self.step_strategy == "cma_csa":
            self._adapt_parameters_cma_csa(Q_bounded)
        elif self.step_strategy == "spectral_cosine":
            self._adapt_parameters_spectral_cosine(t_cur, t_max)

        # 4. Underdamped-Langevin momentum (M3)
        if self.use_momentum:
            if self.V is None or self.V.shape != (N, n):
                self.V = -Q_bounded.copy()
            else:
                gain = (1.0 + self.momentum_beta) if self.nesterov_lookahead else (1.0 - self.momentum_beta)
                self.V = self.momentum_beta * self.V + gain * (-Q_bounded)
            drift = self.V
        else:
            drift = -Q_bounded

        # 4b. Tangential momentum projection (optional)
        if self.momentum_tangential and F.shape[1] > 0:
            m = F.shape[1]
            reg_eye = 1e-7 * np.eye(m)
            Gram = jacobians @ np.swapaxes(jacobians, 1, 2) + reg_eye
            JV = np.einsum("bmn,bn->bm", jacobians, drift)
            try:
                w = np.linalg.solve(Gram, JV[:, :, None])[:, :, 0]
                drift = drift - np.einsum("bmn,bm->bn", jacobians, w)
            except Exception:
                pass

        # 5. Tournament Mating Selection (M4)
        ranks = self.nds.do(F)
        rank_arr = np.zeros(N, dtype=int)
        for r_idx, front in enumerate(ranks):
            rank_arr[front] = r_idx

        n_parents = n_offspring if n_offspring % 2 == 0 else n_offspring + 1
        draws = self.random_state.integers(0, N, size=(2, n_parents))
        parents = np.where(rank_arr[draws[0]] <= rank_arr[draws[1]], draws[0], draws[1])

        # 6. Offspring generation via SBX + Polynomial Mutation (M4)
        ga_offspring = OperatorGA(self.problem, self.pop[parents], Parameter=[1.0, 20.0, 1.0, 20.0], rng=self.random_state)
        off_X = np.asarray(ga_offspring.get("X"), dtype=float)[:n_offspring]

        # 7. Stochastic Gradient Drift Guidance along Langevin field (Canonical SDE Euler-Maruyama)
        drift_parent = self.V[parents[:n_offspring]]
        noise = self.random_state.standard_normal((n_offspring, n))
        effective_noise = self.epsilon * math.sqrt(max(1e-12, self.sigma)) * noise * box
        next_X = off_X + self.sigma * drift_parent * box + effective_noise
        if self.problem.has_bounds():
            next_X = np.clip(next_X, self.problem.xl, self.problem.xu)
        self.pop.set("V", self.V)
        infills = Population.new("X", next_X)
        infills.set("V", drift_parent)
        return infills

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            return

        # Combine parent and offspring populations (2N individuals)
        combined = Population.merge(self.pop, infills)
        self._update_reference_bounds(combined.get("F"))

        # Update success rate only when Strategy 2 is actively requested
        if self.step_strategy == "success_rule":
            ranks = self.nds.do(combined.get("F"))
            n_offspring_elite = sum(1 for idx in ranks[0] if idx >= len(self.pop))
            self._adapt_parameters_success_rule(n_offspring_elite, len(infills))
        # NSGA-III Reference-Direction Niching Survival Selection (2N -> N) on MUD (M6)
        self.pop = self.survival.do(self.problem, combined, n_survive=self.pop_size)

        # Synchronize velocity vector V from surviving individuals
        v_raw = self.pop.get("V")
        if v_raw is not None and len(v_raw) == len(self.pop):
            self.V = np.asarray(v_raw, dtype=float)
        else:
            self.V = np.zeros((len(self.pop), self.problem.n_var), dtype=float)
        self.opt = self.pop

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
        self.opt = self.pop


# Aliases for canonical naming and backward compatibility
GSA_SSW_MUD = GSASSWMUD
SSW2 = GSASSWMUD
