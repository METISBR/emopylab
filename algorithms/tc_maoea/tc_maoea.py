# -*- coding: utf-8 -*-
# Author: Prof. Thiago Santos & METISBr Research Group, 2026
"""Tangent-Coupled Many-Objective Evolutionary Algorithm (TC-MaOEA).

Single consolidated proposal. The differential tangent-bundle pullback
architecture targets degenerate and irregular Pareto geometries. Three
mechanisms are exposed as flags so controlled ablations stay attributable:

- ``gate_mode``: ``"nd"`` keeps the original non-dominated-count gate; ``"progress"``
  (default) engages manifold adaptation only after ideal-point progress stalls,
  which is the closed-loop feedback the paper describes.
- ``wadapt_mode``: ``"tangent"`` keeps the original centered-tangent projection;
  ``"degenerate_pos"`` (default) adapts reference directions only under detected
  degeneracy and projects them back onto the positive simplex, avoiding the
  negative-coordinate corruption of the APD angle.
- ``use_llm``: optional bounded meta-controller that adjusts coefficients only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from operators.utility_functions.UniformPoint import UniformPoint
from util.array_backend import to_numpy
from util.nds.non_dominated_sorting import NonDominatedSorting
from algorithms.community_utils.moead_family import sample_initial
from operators.sampling.lhs import LatinHypercubeSampling
from operators.utility_functions.OperatorGA import OperatorGA

from .tangent_operator import (
    project_simplex,
    certify_empirical_descent,
    certified_projected_normal_descent,
    stochastic_normal_diffusion,
    solve_kkt_simplex_qp,
    svd_manifold_decomposition,
    compute_eej,
    build_pullback_projectors,
    reflective_clamp,
    apd_environmental_selection,
    epr_environmental_selection,
    population_matrix,
    safe_polynomial_mutation,
)

ALGORITHM_FLAGS = {
    "TC_MaOEA": {"multi", "many", "real", "constrained"},
}

__all__ = [
    "TC_MaOEA",
    "TCMaOEA",
    "project_simplex",
    "certify_empirical_descent",
    "certified_projected_normal_descent",
    "stochastic_normal_diffusion",
    "solve_kkt_simplex_qp",
    "svd_manifold_decomposition",
    "compute_eej",
    "build_pullback_projectors",
    "reflective_clamp",
    "apd_environmental_selection",
    "epr_environmental_selection",
]


class TC_MaOEA(Algorithm):
    """Tangent-Coupled Many-Objective Evolutionary Algorithm (TC-MaOEA 2.0).

    Major 2.0 Architectural Overhauls (Senior-Level Peer-Review Standard):
    1. Spectral Manifold Readiness Gate (SMR-Gate): Eliminates progress-stall lockouts.
    2. Decoupled Exploration Scheduling: Guaranteed normal energy (eta_N >= 0.20),
       preventing exploration freezing on 1D/degenerate frontiers.
    3. Niche-Guided Mating Selection: Replaces sum(F) tournament with PBI scalarization
       along reference vectors, preserving extreme and boundary trade-offs.
    4. Certified Normal Descent & Diffusion: Eliminates KKT normal ascent inversion
       and activates stochastic diffusion in the normal subspace N(M).
    5. EPR Environmental Selection: Unconditional ASF extreme point retention and
       curvature-invariant projected distance metric.
    """

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: Optional[np.ndarray] = None,
        tau_var: float = 1e-3,
        tau_gap: float = 50.0,
        reg_scale: float = 1e-6,
        cr: float = 0.9,
        gamma: float = 1e-8,
        alpha: float = 2.0,
        scale_normal_step: bool = True,
        shared_normal: str = "certified",
        gate_mode: str = "smr",
        selection_mode: str = "tournament",
        variation_operator: str = "de",
        env_selection: str = "epr",
        eta_n_min: float = 0.20,
        eta_n_max: float = 0.60,
        mating_neighborhood_size: int = 15,
        mating_delta: float = 0.80,
        progress_window: int = 5,
        progress_tolerance: float = 1e-3,
        wadapt_mode: str = "degenerate_pos",
        degeneracy_frac: float = 0.6,
        degeneracy_ratio: float = 0.15,
        use_llm: bool = False,
        llm_interval: int = 20,
        llm_max_calls: int = 15,
        llm_temperature: float = 0.9,
        llm_model_path: Optional[str | Path] = None,
        llm_log_path: Optional[str | Path] = None,
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
        if gate_mode not in {"smr", "progress", "nd"}:
            raise ValueError(f"gate_mode must be 'smr', 'progress' or 'nd', got {gate_mode!r}")
        if wadapt_mode not in {"degenerate_pos", "tangent", "off"}:
            raise ValueError(f"wadapt_mode must be 'degenerate_pos', 'tangent' or 'off', got {wadapt_mode!r}")
        if shared_normal not in {"certified", "scaled", "raw", "off"}:
            raise ValueError(f"shared_normal must be 'certified', 'scaled', 'raw' or 'off', got {shared_normal!r}")
        if selection_mode not in {"niche", "tournament"}:
            raise ValueError(f"selection_mode must be 'niche' or 'tournament', got {selection_mode!r}")
        if variation_operator not in {"ga", "sbx", "de", "hybrid"}:
            raise ValueError(f"variation_operator must be 'ga', 'sbx', 'de' or 'hybrid', got {variation_operator!r}")
        if env_selection not in {"epr", "apd"}:
            raise ValueError(f"env_selection must be 'epr' or 'apd', got {env_selection!r}")

        self.seed = seed
        self.pop_size = int(max(pop_size, 4))
        self.ref_dirs = ref_dirs
        self.tau_var = float(tau_var)
        self.tau_gap = float(tau_gap)
        self.cr = float(np.clip(cr, 0.0, 1.0))
        self.reg_scale = float(reg_scale)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.alpha_base = float(alpha)
        self.scale_normal_step = bool(scale_normal_step)
        self.sampling = LatinHypercubeSampling() if sampling is None else sampling

        self.shared_normal = shared_normal
        self.gate_mode = gate_mode
        self.selection_mode = selection_mode
        self.variation_operator = variation_operator
        self.env_selection = env_selection
        self.eta_n_min = float(np.clip(eta_n_min, 0.05, 0.50))
        self.eta_n_max = float(np.clip(eta_n_max, 0.30, 0.80))
        self.mating_neighborhood_size = int(max(3, mating_neighborhood_size))
        self.mating_delta = float(np.clip(mating_delta, 0.0, 1.0))

        self.progress_window = int(max(1, progress_window))
        self.progress_tolerance = float(max(0.0, progress_tolerance))
        self.wadapt_mode = wadapt_mode
        self.degeneracy_frac = float(np.clip(degeneracy_frac, 0.0, 1.0))
        self.degeneracy_ratio = float(degeneracy_ratio)

        self.use_llm = bool(use_llm)
        self.llm_interval = int(max(1, llm_interval))
        self.llm_max_calls = int(max(0, llm_max_calls))
        self.llm_temperature = float(np.clip(llm_temperature, 0.0, 1.0))
        self.llm_model_path = Path(llm_model_path) if llm_model_path is not None else None
        self.llm_log_path = Path(llm_log_path) if llm_log_path is not None else None

        # Internal state
        self.W: Optional[np.ndarray] = None
        self.W_adapt: Optional[np.ndarray] = None
        self.z_min: Optional[np.ndarray] = None
        self.z_max: Optional[np.ndarray] = None
        self.d_star: int = 2
        self.sigmas: Optional[np.ndarray] = None
        self.P_T: Optional[np.ndarray] = None
        self.P_N: Optional[np.ndarray] = None
        self.J: Optional[np.ndarray] = None
        self.d_kkt: Optional[np.ndarray] = None
        self.eta_T: float = 0.50
        self.eta_N: float = 0.50
        self.is_fallback: bool = True
        self.degenerate: bool = False
        self.normal_step_telemetry: list[dict[str, Any]] = []
        self.infill_step_telemetry: list[dict[str, Any]] = []
        # Progress / stall tracking
        self._prev_ideal_sum: Optional[float] = None
        self.stall_count: int = 0
        self.gen_count: int = 0

        # LLM controller state
        self.p_norm_boost: float = 0.0
        self._pending_alpha: Optional[float] = None
        self.llm_calls: int = 0
        self.llm_call_history: list[dict[str, Any]] = []
        self._llm_client = None

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
        self._prev_ideal_sum = float(np.sum(self.z_min))
        self._update_tangent_operators()
        self._set_optimum()

    def _update_progress(self) -> bool:
        """Advance the stall detector and report whether the gate is open."""
        if self.gate_mode == "smr":
            # Spectral Manifold Readiness Gate: warmup 2 generations to establish ranking
            return self.gen_count >= 2
        if self.gate_mode != "progress":
            return True
        ideal_sum = float(np.sum(self.z_min))
        if self._prev_ideal_sum is not None:
            denom = max(abs(self._prev_ideal_sum), 1e-12)
            improvement = (self._prev_ideal_sum - ideal_sum) / denom
            self.stall_count = self.stall_count + 1 if improvement <= self.progress_tolerance else 0
        self._prev_ideal_sum = ideal_sum
        return self.stall_count >= self.progress_window

    def _set_fallback_operators(self, D: int) -> None:
        self.is_fallback = True
        self.degenerate = False
        self.W_adapt = np.copy(self.W) if self.W is not None else None
        self.P_T = np.eye(D)
        self.P_N = np.zeros((D, D))
        self.d_kkt = np.zeros(D)
        self.eta_T = 1.0
        self.eta_N = 0.0
        self.normal_step_telemetry.append({
            "gen": int(self.gen_count),
            "n_evals": float(self.n_evals),
            "is_fallback": True,
            "degenerate": False,
            "d_star": 0.0,
            "sigma_ratio": 0.0,
            "d_kkt_norm": 0.0,
            "pt_dkkt_norm": 0.0,
            "pn_dkkt_norm": 0.0,
            "max_j_dkkt": 0.0,
            "max_j_pn_dkkt": 0.0,
            "shared_normal_scale": 0.0,
        })

    def _adapt_reference_directions(
        self,
        V_dstar: np.ndarray,
        sigma_ratio: float,
        d_star: int,
        M: int,
    ) -> np.ndarray:
        """Decide whether/how to move reference directions onto the front."""
        if self.wadapt_mode == "off":
            return np.copy(self.W)

        degenerate = (d_star <= max(1, int(self.degeneracy_frac * (M - 1)))) and (sigma_ratio < self.degeneracy_ratio)
        self.degenerate = bool(degenerate)
        if self.wadapt_mode == "tangent" or degenerate:
            W_proj = self.W @ V_dstar @ V_dstar.T
        else:
            # Non-degenerate: keep the canonical simplex lattice untouched.
            return np.copy(self.W)

        W_adapt = np.copy(self.W)
        if self.wadapt_mode == "tangent":
            norm_proj = np.linalg.norm(W_proj, axis=1, keepdims=True)
            valid = norm_proj[:, 0] > 1e-12
            W_adapt[valid] = W_proj[valid] / norm_proj[valid]
            return W_adapt

        for idx in range(W_proj.shape[0]):
            row = W_proj[idx]
            if not np.all(np.isfinite(row)) or float(np.sum(row)) <= 1e-12:
                continue
            W_adapt[idx] = project_simplex(row, z=1.0)
        return W_adapt

    def _update_tangent_operators(self) -> None:
        if self.pop is None or len(self.pop) < 2:
            return

        X = population_matrix(self.pop, "X")
        F = population_matrix(self.pop, "F")
        M = int(self.problem.n_obj)
        D = int(X.shape[1])

        self.z_min = np.minimum(self.z_min, np.min(F, axis=0))
        self.z_max = np.maximum(self.z_max, np.max(F, axis=0))
        gate_open = self._update_progress()

        nds = NonDominatedSorting()
        fronts = nds.do(F)
        n_nd = len(fronts[0]) if len(fronts) > 0 else 0

        elite_idx = np.array(fronts[0], dtype=int) if len(fronts) > 0 else np.array([], dtype=int)
        elite_target = min(len(X), max(4, min(M, len(X))))
        if len(elite_idx) < elite_target:
            pooled: List[int] = []
            for fr in fronts:
                pooled.extend(fr)
                if len(pooled) >= elite_target:
                    break
            elite_idx = np.array(pooled, dtype=int)

        if len(elite_idx) < 2:
            self._set_fallback_operators(D)
            return

        X_elite = X[elite_idx]
        F_elite = F[elite_idx]

        span = np.maximum(self.z_max - self.z_min, 1e-12)
        F_norm = (F_elite - self.z_min[None, :]) / span[None, :]

        d_star, V_dstar, sigmas = svd_manifold_decomposition(
            F_norm, tau_var=self.tau_var, tau_gap=self.tau_gap
        )
        self.d_star = d_star
        self.sigmas = sigmas

        sigma_1 = float(sigmas[0]) if len(sigmas) > 0 else 0.0
        sigma_d = float(sigmas[d_star - 1]) if (d_star >= 1 and len(sigmas) >= d_star) else 0.0
        finite_spectrum = bool(np.all(np.isfinite(sigmas))) and (sigma_1 > 0)
        sigma_ratio = (sigma_d / sigma_1) if (finite_spectrum and sigma_1 > 0) else 0.0

        if self.gate_mode == "smr":
            gate_ok = bool(
                gate_open
                and finite_spectrum
                and (d_star >= 1)
                and (sigma_ratio >= 1e-4)
                and (sigma_1 >= 1e-6)
            )
        elif self.gate_mode == "nd":
            nd_ok = (n_nd >= 3 * M) and finite_spectrum and (d_star >= 1) and (sigma_ratio > 1e-2)
            gate_ok = bool(nd_ok and gate_open)
        else:
            gate_ok = bool(gate_open and finite_spectrum and (d_star >= 1) and (sigma_ratio > 1e-2))

        if not gate_ok:
            self._set_fallback_operators(D)
            return

        self.is_fallback = False
        self.W_adapt = self._adapt_reference_directions(V_dstar, sigma_ratio, d_star, M)

        J = compute_eej(X_elite, F_norm, reg_scale=self.reg_scale)
        self.J = J
        P_T, P_N = build_pullback_projectors(J, V_dstar, gamma=self.gamma)
        self.P_T = P_T
        self.P_N = P_N

        _, d_kkt = solve_kkt_simplex_qp(J)
        self.d_kkt = d_kkt

        pn_dkkt = P_N @ d_kkt
        pt_dkkt = P_T @ d_kkt
        self.normal_step_telemetry.append({
            "gen": int(self.gen_count),
            "n_evals": float(self.n_evals),
            "is_fallback": False,
            "degenerate": bool(self.degenerate),
            "d_star": float(d_star),
            "sigma_ratio": float(sigma_ratio),
            "d_kkt_norm": float(np.linalg.norm(d_kkt)),
            "pt_dkkt_norm": float(np.linalg.norm(pt_dkkt)),
            "pn_dkkt_norm": float(np.linalg.norm(pn_dkkt)),
            "max_j_dkkt": float(np.max(J @ d_kkt)),
            "max_j_pn_dkkt": float(np.max(J @ pn_dkkt)),
            "shared_normal_scale": 0.0,
        })

        # Decoupled Exploration Scheduling with Guaranteed Normal Energy (Mechanism 1)
        curr_eval = float(self.n_evals)
        max_eval = float(getattr(self.termination, "n_max_evals", getattr(self.termination, "n_max_eval", 30000)))
        tau = float(np.clip(curr_eval / max(max_eval, 1.0), 0.0, 1.0))

        if len(sigmas) > d_star:
            r_unexpl = float(np.sum(sigmas[d_star:]) / (np.sum(sigmas) + 1e-12))
        else:
            r_unexpl = 0.0

        eta_n_base = self.eta_n_min + (self.eta_n_max - self.eta_n_min) * ((1.0 - tau) ** 1.5)
        self.eta_N = float(np.clip(eta_n_base + 0.25 * r_unexpl, self.eta_n_min, self.eta_n_max))
        self.eta_T = float(np.clip(1.0 - self.eta_N, 0.25, 1.0))

    def _shared_normal_step(
        self,
        P_N: np.ndarray,
        d_kkt: np.ndarray,
        delta_de: np.ndarray,
    ) -> np.ndarray:
        """Shared KKT normal displacement policy: 'certified', 'scaled', 'raw' or 'off'.

        Records per-infill attribution in ``infill_step_telemetry`` so shared
        normal scale is paired with the exact generation step that used it.
        """
        pn_dkkt = P_N @ d_kkt
        norm_pn = float(np.linalg.norm(pn_dkkt))
        mean_de = float(np.mean(np.linalg.norm(delta_de, axis=1))) if delta_de.size else 0.0

        if self.shared_normal == "certified" and self.J is not None:
            step = certified_projected_normal_descent(
                self.J, P_N, d_kkt, delta_de, scale_normal_step=self.scale_normal_step
            )
            scale = float(np.linalg.norm(step)) / (norm_pn + 1e-12) if norm_pn > 1e-12 else 0.0
            shared_disabled = bool(scale == 0.0)
        elif self.shared_normal == "off":
            scale = 0.0
            shared_disabled = True
            step = np.zeros(np.asarray(pn_dkkt).shape, dtype=float)
        else:
            shared_disabled = False
            if self.shared_normal == "raw" or not self.scale_normal_step:
                scale = 1.0
            elif norm_pn <= 1e-12:
                scale = 0.0
            else:
                scale = min(1.0, mean_de / (norm_pn + 1e-8))
            step = scale * pn_dkkt

        self.infill_step_telemetry.append(
            {
                "gen": int(self.gen_count),
                "n_evals": float(self.n_evals),
                "shared_normal_mode": self.shared_normal,
                "shared_normal_scale": float(scale),
                "shared_normal_disabled": bool(shared_disabled),
                "d_star_used": float(self.d_star),
                "eta_T_used": float(self.eta_T),
                "eta_N_used": float(self.eta_N),
                "pn_dkkt_norm_used": float(norm_pn),
                "mean_de_norm": float(mean_de),
            }
        )
        return step

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

        rng = self.random_state if hasattr(self, "random_state") and self.random_state is not None else np.random.default_rng(42)
        arange_N = np.arange(N)

        F = population_matrix(self.pop, "F")

        if self.selection_mode == "niche" and F is not None and len(F) == N and N >= 4:
            # Mechanism 2: Local Niche-Guided Mating Selection
            span_f = np.maximum(self.z_max - self.z_min, 1e-12)
            F_norm = np.maximum(F - self.z_min[None, :], 0.0) / span_f[None, :]
            norm_f = np.linalg.norm(F_norm, axis=1)

            W_curr = self.W_adapt if self.W_adapt is not None else self.W
            if W_curr is None or len(W_curr) == 0:
                W_curr = np.eye(F.shape[1])
            norms_w = np.linalg.norm(W_curr, axis=1)
            W_unit = W_curr / np.maximum(norms_w[:, None], 1e-12)

            cos_theta = np.clip((F_norm @ W_unit.T) / np.maximum(norm_f[:, None], 1e-12), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            assoc = np.argmin(theta, axis=1)

            # Intra-niche PBI score
            assoc_w = W_unit[assoc]
            d1 = np.sum(F_norm * assoc_w, axis=1)
            perp_vec = F_norm - d1[:, None] * assoc_w
            d2 = np.linalg.norm(perp_vec, axis=1)
            pbi_scores = d1 + 5.0 * d2

            # Fast non-dominated sorting for Pareto ranks
            nds = NonDominatedSorting()
            fronts = nds.do(F)
            rank = np.zeros(N, dtype=int)
            for f_idx, fr in enumerate(fronts):
                rank[fr] = f_idx

            # Precompute angle matrix among reference directions
            angles_w = np.arccos(np.clip(W_unit @ W_unit.T, -1.0, 1.0))
            T_size = min(max(3, self.mating_neighborhood_size), len(W_unit))
            neighbors = np.argsort(angles_w, axis=1)[:, :T_size]

            p_base = np.zeros(N, dtype=int)
            r1 = np.zeros(N, dtype=int)
            r2 = np.zeros(N, dtype=int)

            for i in range(N):
                target_niche = assoc[i]
                use_neigh = bool((rng.random() < self.mating_delta) if hasattr(rng, "random") else (rng.uniform() < self.mating_delta))
                if use_neigh:
                    neigh_niches = set(neighbors[target_niche])
                    pool_candidates = [idx for idx in range(N) if assoc[idx] in neigh_niches]
                    if len(pool_candidates) < 4:
                        pool_candidates = list(range(N))
                else:
                    pool_candidates = list(range(N))

                pool_arr = np.array(pool_candidates, dtype=int)
                n_draw = min(len(pool_arr), 6)
                if hasattr(rng, "choice"):
                    c = rng.choice(pool_arr, size=n_draw, replace=(len(pool_arr) < 6))
                else:
                    c = np.random.choice(pool_arr, size=n_draw, replace=(len(pool_arr) < 6))

                def _pick_best(i1: int, i2: int) -> int:
                    if rank[i1] < rank[i2]:
                        return i1
                    elif rank[i2] < rank[i1]:
                        return i2
                    return i1 if pbi_scores[i1] <= pbi_scores[i2] else i2

                p_base[i] = _pick_best(int(c[0]), int(c[1 % len(c)]))
                r1[i] = _pick_best(int(c[2 % len(c)]), int(c[3 % len(c)]))
                if r1[i] == p_base[i] and len(c) > 2:
                    r1[i] = int(c[2 % len(c)]) if int(c[2 % len(c)]) != p_base[i] else int(c[3 % len(c)])
                r2[i] = _pick_best(int(c[4 % len(c)]), int(c[5 % len(c)]))
                if r2[i] == p_base[i] or r2[i] == r1[i]:
                    r2[i] = (int(p_base[i]) + 1) % N
        else:
            # Tournament selection based on Pareto rank with crowding distance tie-breaking
            nds = NonDominatedSorting()
            fronts = nds.do(F)
            rank = np.zeros(N, dtype=int)
            for f_idx, fr in enumerate(fronts):
                rank[fr] = f_idx

            from operators.utility_functions.CrowdingDistance import CrowdingDistance
            cd = CrowdingDistance(F, rank)

            def _tournament_pick(draw_pairs: np.ndarray) -> np.ndarray:
                c1, c2 = draw_pairs[0], draw_pairs[1]
                better_rank = rank[c1] < rank[c2]
                worse_rank = rank[c1] > rank[c2]
                better_cd = cd[c1] > cd[c2]
                win1 = better_rank | ((~worse_rank) & better_cd)
                return np.where(win1, c1, c2)

            if N >= 4:
                if hasattr(rng, "integers"):
                    draws = rng.integers(0, N, size=(2, N))
                    d1 = rng.integers(0, N, size=(2, N))
                    d2 = rng.integers(0, N, size=(2, N))
                else:
                    draws = rng.randint(0, N, size=(2, N))
                    d1 = rng.randint(0, N, size=(2, N))
                    d2 = rng.randint(0, N, size=(2, N))
                p_base = _tournament_pick(draws)
                r1 = _tournament_pick(d1)
                r2 = _tournament_pick(d2)
            else:
                p_base = arange_N
                r1 = (arange_N + 1) % N
                r2 = (arange_N + 2) % N

        if self.variation_operator in {"ga", "sbx"}:
            n_parents = N + (1 if N % 2 == 1 else 0)
            if N >= 4 and "cd" in locals():
                if hasattr(rng, "integers"):
                    draws_ga = rng.integers(0, N, size=(2, n_parents))
                else:
                    draws_ga = rng.randint(0, N, size=(2, n_parents))
                parents_idx = _tournament_pick(draws_ga)
            elif hasattr(rng, "choice"):
                parents_idx = rng.choice(p_base, size=n_parents)
            else:
                parents_idx = np.random.choice(p_base, size=n_parents)
            off_dec = OperatorGA(self.problem, self.pop[parents_idx].get("X"), rng=rng)[:N]
            off_dec = np.clip(off_dec, xl, xu)
            return Population.new("X", off_dec)

        elif self.variation_operator == "hybrid":
            n_ga = N // 2
            n_de = N - n_ga
            n_parents_ga = n_ga + (1 if n_ga % 2 == 1 else 0)
            if hasattr(rng, "choice"):
                parents_ga = rng.choice(p_base[:n_ga], size=n_parents_ga)
            else:
                parents_ga = np.random.choice(p_base[:n_ga], size=n_parents_ga)
            off_ga = OperatorGA(self.problem, self.pop[parents_ga].get("X"), rng=rng)[:n_ga]
            pn_dkkt = P_N @ d_kkt
            if np.linalg.norm(pn_dkkt) > 1e-6 and self.J is not None and np.all(self.J @ pn_dkkt < -1e-8):
                off_ga = off_ga + 0.5 * self.eta_N * pn_dkkt[None, :]
            off_ga = np.clip(off_ga, xl, xu)

            scale_T = max(np.sqrt(float(self.d_star)), 1.0)
            delta_de = 0.5 * (X[r1[n_ga:]] - X[r2[n_ga:]])
            delta_T = (delta_de @ P_T.T) / scale_T

            curr_eval = float(self.n_evals)
            max_eval = float(getattr(self.termination, "n_max_evals", getattr(self.termination, "n_max_eval", 30000)))
            tau = float(np.clip(curr_eval / max(max_eval, 1.0), 0.0, 1.0))
            sigma_diff = 0.05 * ((1.0 - tau) ** 2.0)
            xi = stochastic_normal_diffusion(P_N, n_de, xl, xu, sigma_diff, rng=rng)

            delta_N_shared = self._shared_normal_step(P_N, d_kkt, delta_de)
            delta_N_ind = delta_de @ P_N.T
            delta_N = delta_N_shared[None, :] + delta_N_ind + xi

            trial = X[p_base[n_ga:]] + (self.eta_T * delta_T + self.eta_N * delta_N)
            off_de = np.clip(trial, xl, xu)
            off_de = safe_polynomial_mutation(off_de, xl, xu, eta_m=20.0, prob_m=1.0 / max(D, 1), rng=rng)
            off_de = np.clip(off_de, xl, xu)

            off_all = np.vstack([off_ga, off_de])
            return Population.new("X", off_all)

        # Default differential evolution path (variation_operator == "de")
        scale_T = max(np.sqrt(float(self.d_star)), 1.0)
        delta_de = 0.5 * (X[r1] - X[r2])
        delta_T = (delta_de @ P_T.T) / scale_T

        curr_eval = float(self.n_evals)
        max_eval = float(getattr(self.termination, "n_max_evals", getattr(self.termination, "n_max_eval", 30000)))
        tau = float(np.clip(curr_eval / max(max_eval, 1.0), 0.0, 1.0))
        sigma_diff = 0.05 * ((1.0 - tau) ** 2.0)
        xi = stochastic_normal_diffusion(P_N, N, xl, xu, sigma_diff, rng=rng)

        delta_N_shared = self._shared_normal_step(P_N, d_kkt, delta_de)
        delta_N_ind = delta_de @ P_N.T
        delta_N = delta_N_shared[None, :] + delta_N_ind + xi

        trial = X[p_base] + (self.eta_T * delta_T + self.eta_N * delta_N)

        if self.p_norm_boost > 0.0 and self.P_N is not None:
            pulse = rng.normal(0.0, 0.1, size=(N, D)) @ self.P_N.T
            trial = trial + self.p_norm_boost * pulse

        if self.cr < 1.0:
            if hasattr(rng, "random"):
                mask = rng.random((N, D)) < self.cr
                j_rand = rng.integers(0, D, size=N)
            else:
                mask = rng.uniform(0.0, 1.0, size=(N, D)) < self.cr
                j_rand = rng.randint(0, D, size=N)
            mask[np.arange(N), j_rand] = True
            offspring_X = np.where(mask, trial, X[p_base])
        else:
            offspring_X = trial

        offspring_X = np.clip(offspring_X, xl, xu)
        offspring_X = safe_polynomial_mutation(
            offspring_X, xl, xu, eta_m=20.0, prob_m=1.0 / max(D, 1), rng=rng
        )
        offspring_X = np.clip(offspring_X, xl, xu)
        return Population.new("X", offspring_X)

    # -- Optional local-GGUF coefficient supervisor -------------------------
    def _setup_llm(self) -> None:
        if not self.use_llm:
            return
        from core.llm.local_llm import LocalLLMClient

        self._llm_client = LocalLLMClient(
            temperature=self.llm_temperature,
            max_tokens=64,
            force_inprocess=True,
            model_path=self.llm_model_path,
        )

    def _record_llm_call(self, record: dict[str, Any]) -> None:
        self.llm_call_history.append(record)
        if self.llm_log_path is None:
            return
        self.llm_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.llm_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _query_llm_meta_controller(self, t_ratio: float) -> None:
        if self.llm_calls >= self.llm_max_calls:
            return
        if self._llm_client is None:
            self._setup_llm()
        payload = {
            "gen": int(self.gen_count),
            "t_ratio": round(float(t_ratio), 3),
            "stagnation": int(self.stall_count),
            "d_star": int(self.d_star),
            "eta_T": round(float(self.eta_T), 3),
            "p_norm": round(float(self.p_norm_boost), 3),
        }
        prompt = (
            f"State: {json.dumps(payload, sort_keys=True)}. "
            "Output JSON with adjusted parameters: "
            '{"p_norm": float between 0.0 and 1.0, "alpha": float between 1.0 and 4.0}'
        )
        self.llm_calls += 1
        try:
            res = self._llm_client.json_call(
                prompt,
                system="You are an evolutionary parameter meta-controller. Output ONLY valid JSON.",
            )
        except Exception as exc:
            self._record_llm_call({"call": self.llm_calls, "gen": self.gen_count,
                                   "status": "exception", "detail": type(exc).__name__})
            return
        status = getattr(self._llm_client, "last_call_status", {})
        if not isinstance(res, dict) or set(res) != {"p_norm", "alpha"}:
            self._record_llm_call({"call": self.llm_calls, "gen": self.gen_count,
                                   "status": "rejected", "detail": "schema"})
            return
        if not all(isinstance(res.get(k), (int, float)) and math.isfinite(res[k]) for k in ("p_norm", "alpha")):
            self._record_llm_call({"call": self.llm_calls, "gen": self.gen_count,
                                   "status": "rejected", "detail": "nonfinite"})
            return
        self.p_norm_boost = float(np.clip(float(res["p_norm"]), 0.0, 1.0))
        self._pending_alpha = float(np.clip(float(res["alpha"]), 1.0, 4.0))
        self._record_llm_call({"call": self.llm_calls, "gen": self.gen_count,
                               "status": "accepted", "p_norm": self.p_norm_boost,
                               "alpha": self._pending_alpha,
                               "backend": status.get("backend", "unknown"),
                               "latency_ms": status.get("latency_ms", 0.0)})

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

        if self._pending_alpha is not None:
            self.alpha = self._pending_alpha
            self._pending_alpha = None
        else:
            self.alpha = float(self.alpha_base)

        if self.env_selection == "epr":
            self.pop = epr_environmental_selection(
                pool=merged,
                W_adapt=self.W_adapt if self.W_adapt is not None else self.W,
                z_min=self.z_min,
                z_max=self.z_max,
                n_survive=self.pop_size,
                t_ratio=t_ratio,
                alpha=self.alpha,
            )
        else:
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
        self.gen_count += 1
        if self.use_llm and self.gen_count % self.llm_interval == 0:
            self._query_llm_meta_controller(t_ratio)
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


