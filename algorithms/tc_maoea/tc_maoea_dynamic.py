# -*- coding: utf-8 -*-
"""EXPERIMENTAL - Adaptive Tangent-Coupled MaOEA (TC-MaOEA-Dynamic).

Frozen research scaffolding. Extends the foundational TC-MaOEA with a
closed-loop dynamic meta-controller:

1. Monitors generational stagnation via best-sum progress.
2. Injects orthogonal normal pulse p_N = P_N @ xi (xi ~ N(0, sigma^2))
   subject to the computed-subspace identity P_T @ p_N = 0, which concerns
   the estimated projectors only and does not preserve a nonlinear manifold.
3. Modulates APD penalty exponent alpha_t from 1.0 to 4.0.
4. Supports optional local LLM policy supervisor with deterministic fallback.

The LLM hook is inactive by default (use_llm=False) and never calls providers
unless explicitly enabled. Do not wire into TC_MaOEA base path.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional
import numpy as np

from core.population import Population
from .tangent_operator import reflective_clamp
from .tc_maoea import TC_MaOEA

ALGORITHM_FLAGS = {
    "TC_MaOEA_Dynamic": {"multi", "many", "real", "constrained"},
    "TCMaOEADynamic": {"multi", "many", "real", "constrained"},
}


class TC_MaOEA_Dynamic(TC_MaOEA):
    """EXPERIMENTAL - Dynamic Tangent-Coupled MaOEA with orthogonal pulse & LLM hook.

    Parameters:
        use_llm: Whether to query LocalLLMClient for parameter supervision (default: False).
        llm_interval: Generation interval between LLM meta-controller queries (default: 20).
        stag_threshold: Progress threshold below which stagnation triggers (default: 1e-4).
        max_stag_window: Consecutive stagnation generations before pulse activation (default: 5).
        pulse_strength: Base std scaling for the orthogonal normal pulse (default: 0.1).
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_interval: int = 20,
        stag_threshold: float = 1e-4,
        max_stag_window: int = 5,
        pulse_strength: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_llm = bool(use_llm)
        self.llm_interval = int(max(1, llm_interval))
        self.stag_threshold = float(stag_threshold)
        self.max_stag_window = int(max(1, max_stag_window))
        self.pulse_strength = float(pulse_strength)

        # Dynamic meta-controller state
        self.p_norm_boost: float = 0.0
        self.alpha_dynamic: float = self.alpha
        self.stag_count: int = 0
        self.prev_best_sum: float = math.inf
        self.gen_count: int = 0
        self._llm_client = None

    def _setup_llm(self) -> None:
        if not self.use_llm:
            return
        try:
            from core.llm.local_llm import LocalLLMClient
            self._llm_client = LocalLLMClient(temperature=0.2, max_tokens=64)
        except Exception:
            self._llm_client = None

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> Any:
        if infills is None or len(infills) == 0:
            return

        # Schedule the dynamic APD exponent before selection so this
        # generation's survival uses it (not the stale base alpha).
        curr_eval = float(self.n_evals)
        max_eval = float(getattr(self.termination, "n_max_evals", getattr(self.termination, "n_max_eval", 30000)))
        t_ratio = float(np.clip(curr_eval / max(max_eval, 1.0), 0.0, 1.0))
        self.alpha_dynamic = float(np.clip(1.0 + 2.5 * (t_ratio ** 1.5), 1.0, 4.0))
        self.alpha = self.alpha_dynamic

        super()._advance(infills=infills, **kwargs)

        # Stagnation monitoring
        from .tangent_operator import population_matrix
        F_curr = population_matrix(self.pop, "F")
        curr_best_sum = float(np.min(np.sum(F_curr, axis=1)))
        prev = self.prev_best_sum
        delta = prev - curr_best_sum if math.isfinite(prev) else math.inf

        if delta <= self.stag_threshold:
            self.stag_count += 1
            if self.stag_count >= self.max_stag_window:
                self.p_norm_boost = float(np.clip(self.p_norm_boost + 0.2, 0.0, 1.0))
        else:
            self.stag_count = 0
            self.p_norm_boost = float(np.clip(self.p_norm_boost - 0.1, 0.0, 1.0))

        self.prev_best_sum = curr_best_sum
        self.gen_count += 1

        if self.use_llm and (self.gen_count % self.llm_interval == 0):
            self._query_llm_meta_controller(t_ratio)

    def _query_llm_meta_controller(self, t_ratio: float) -> None:
        """Queries local LLM policy supervisor with strict accept-or-ignore policy."""
        if self._llm_client is None:
            self._setup_llm()
        if self._llm_client is None:
            return

        payload = {
            "gen": self.gen_count,
            "t_ratio": round(t_ratio, 3),
            "stagnation": self.stag_count,
            "d_star": int(self.d_star),
            "eta_T": round(float(self.eta_T), 3),
            "p_norm": round(float(self.p_norm_boost), 3),
        }
        prompt = (
            f"State: {json.dumps(payload)}. "
            "Output JSON with adjusted parameters: "
            '{"p_norm": float between 0.0 and 1.0, "alpha": float between 1.0 and 4.0}'
        )
        try:
            res = self._llm_client.json_call(
                prompt,
                system="You are an evolutionary parameter meta-controller. Output ONLY valid JSON.",
            )
        except Exception:
            return  # Fail closed to deterministic state
        if not isinstance(res, dict) or set(res) - {"p_norm", "alpha"}:
            return  # Ignore NaN/partial/unexpected payloads; never partially update
        if not all(isinstance(res.get(k), (int, float)) and math.isfinite(res[k]) for k in ("p_norm", "alpha")):
            return
        self.p_norm_boost = float(np.clip(float(res["p_norm"]), 0.0, 1.0))
        self.alpha_dynamic = float(np.clip(float(res["alpha"]), 1.0, 4.0))

    def _compute_delta_x(self, delta_T: np.ndarray, delta_N: np.ndarray, N: int, D: int) -> np.ndarray:
        delta_x = super()._compute_delta_x(delta_T, delta_N, N, D)
        if self.p_norm_boost > 0.0:
            P_N = self.P_N
            xi = self.random_state.normal(0, self.pulse_strength, size=(N, D))
            pulse_N = xi @ P_N.T
            delta_x = delta_x + self.p_norm_boost * pulse_N
        return delta_x


TCMaOEADynamic = TC_MaOEA_Dynamic
