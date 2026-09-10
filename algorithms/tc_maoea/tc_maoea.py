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
    safe_polynomial_mutation,
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
        ref_dirs: Explicit reference directions of shape (K, M).
        tau_var / tau_gap: SVD dimension-estimation thresholds.
        reg_scale / gamma: ridge / damped-inverse regularization.
        cr: DE crossover rate.
        alpha: APD penalty exponent.
        scale_normal_step: scale the shared normal step by mean DE displacement.
        early_normal_cap: cap eta_N early in the run.
        gate_mode: "progress" (default) or "nd".
        progress_window / progress_tolerance: stall detector for "progress".
        wadapt_mode: "degenerate_pos" (default) or "tangent".
        degeneracy_frac / degeneracy_ratio: degeneracy detection thresholds.
        use_llm and llm_*: optional bounded local-GGUF coefficient supervisor.
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
        early_normal_cap: bool = False,
        shared_normal: str = "scaled",
        gate_mode: str = "progress",
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
        if gate_mode not in {"progress", "nd"}:
            raise ValueError(f"gate_mode must be 'progress' or 'nd', got {gate_mode!r}")
        if wadapt_mode not in {"degenerate_pos", "tangent", "off"}:
            raise ValueError(f"wadapt_mode must be 'degenerate_pos', 'tangent' or 'off', got {wadapt_mode!r}")

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
        self.early_normal_cap = bool(early_normal_cap)
        self.sampling = sampling

        if shared_normal not in {"scaled", "raw", "off"}:
            raise ValueError(f"shared_normal must be 'scaled', 'raw' or 'off', got {shared_normal!r}")
        self.shared_normal = shared_normal
        self.gate_mode = gate_mode
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
        self.d_kkt: Optional[np.ndarray] = None
        self.eta_T: float = 1.0
        self.eta_N: float = 0.0
        self.is_fallback: bool = True
        self.degenerate: bool = False
        self.normal_step_telemetry: list[dict[str, Any]] = []

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

        if self.gate_mode == "nd":
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
        if delta_N.ndim == 1:
            delta_N = delta_N[None, :]
        delta_x = self.eta_T * delta_T + self.eta_N * delta_N
        if self.p_norm_boost > 0.0 and self.P_N is not None:
            rng = self.random_state if self.random_state is not None else np.random.default_rng(42)
            pulse = rng.normal(0.0, 0.1, size=(N, D)) @ self.P_N.T
            delta_x = delta_x + self.p_norm_boost * pulse
        return delta_x

    def _shared_normal_step(
        self,
        P_N: np.ndarray,
        d_kkt: np.ndarray,
        delta_de: np.ndarray,
    ) -> np.ndarray:
        """Shared KKT normal displacement policy: 'off', 'raw' or 'scaled'.

        'off' is the A2 ablation (no shared normal descent). 'raw' applies the
        certified d_kkt normal component without re-amplification. 'scaled'
        keeps the original mean-DE-matched rescaling.
        """
        if self.shared_normal == "off":
            scale = 0.0
            step = np.zeros(np.asarray(d_kkt).shape, dtype=float)
        else:
            pn_dkkt = P_N @ d_kkt
            norm_pn = float(np.linalg.norm(pn_dkkt))
            if self.shared_normal == "raw" or not self.scale_normal_step:
                scale = 1.0
            elif norm_pn <= 1e-12:
                scale = 0.0
            else:
                mean_de = float(np.mean(np.linalg.norm(delta_de, axis=1)))
                scale = mean_de / (norm_pn + 1e-8)
            step = scale * pn_dkkt
        if self.normal_step_telemetry:
            self.normal_step_telemetry[-1]["shared_normal_scale"] = float(scale)
            self.normal_step_telemetry[-1]["shared_normal_mode"] = self.shared_normal
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
        f_norm = np.sum(F, axis=1) if (F is not None and len(F) == N) else np.arange(N)

        if N >= 4:
            if hasattr(rng, "integers"):
                draws = rng.integers(0, N, size=(2, N))
                d1 = rng.integers(0, N, size=(2, N))
                d2 = rng.integers(0, N, size=(2, N))
            else:
                draws = rng.randint(0, N, size=(2, N))
                d1 = rng.randint(0, N, size=(2, N))
                d2 = rng.randint(0, N, size=(2, N))
            p_base = np.where(f_norm[draws[0]] < f_norm[draws[1]], draws[0], draws[1])
            r1 = np.where(f_norm[d1[0]] < f_norm[d1[1]], d1[0], d1[1])
            r2 = np.where(f_norm[d2[0]] < f_norm[d2[1]], d2[0], d2[1])
        else:
            p_base = arange_N
            r1 = (arange_N + 1) % N
            r2 = (arange_N + 2) % N

        delta_de = 0.5 * (X[r1] - X[r2])
        delta_T = delta_de @ P_T.T
        delta_N_shared = self._shared_normal_step(P_N, d_kkt, delta_de)
        delta_N_ind = delta_de @ P_N.T
        delta_N = delta_N_shared[None, :] + delta_N_ind

        trial = X[p_base] + self._compute_delta_x(delta_T, delta_N, N, D)

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

        offspring_X = reflective_clamp(offspring_X, xl, xu)
        offspring_X = safe_polynomial_mutation(
            offspring_X, xl, xu, eta_m=20.0, prob_m=1.0 / max(D, 1), rng=rng
        )
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
