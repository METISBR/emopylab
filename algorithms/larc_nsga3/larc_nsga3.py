"""LARC-NSGA3: Two-Tier Hybrid Evolutionary Algorithm with Local Quantized LLM Strategy Controller.

Authors
-------
Thiago Santos, Sebastiao Xavier (UFOP / METISBr, 2026).

Paper
-----
"Strict Local-LLM Discrete Action Control for Adaptive NSGA3 in Many-Objective Optimization"
Information Sciences (Elsevier).

Architecture
------------
1. Tier 1 (Deterministic Heuristic): State-aware reactive decision tree executed every generation.
2. Tier 2 (Bounded Semantic Supervisor): Local Qwen2.5-0.5B LLM (via mlx-lm on Apple Silicon or
   llama-cpp-python in-process on Linux/VPS) queried at bounded intervals (B_LLM <= 15).
3. State-Diagnostic Traps: Filters out catastrophic exploration on degenerate/multimodal fronts.
4. Graceful Fallback: In case of endpoint unavailability, parsing failure, or low confidence,
   the controller gracefully falls back to the Tier-1 heuristic action, eliminating survival bias.
5. Zero-Pressure Random Mating: Prevents boundary destruction in high-dimensional spaces (M >= 5).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from core.population import Population

from algorithms.nsga3_local.nsga3_local import (
    NSGA3Local,
    _constraint_violation,
    _environmental_selection,
    _population_constraints,
    _population_objectives,
    _update_zmin,
)
from algorithms.community_utils.moead_family import rng_from_algo
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection

from core.llm.local_llm import (
    DEFAULT_BASE_URL as _DEFAULT_LLM_BASE_URL,
    DEFAULT_MODEL as _DEFAULT_LLM_MODEL,
    LocalLLMClient,
)

logger = logging.getLogger(__name__)

ALGORITHM_FLAGS = {
    "LARC_NSGA3": {"multi", "many", "real", "integer"},
    "Heuristic_NSGA3": {"multi", "many", "real", "integer"},
    "LARC_NSGA3_NoTraps": {"multi", "many", "real", "integer"},
}

_ACTIONS = ("conv", "div", "var", "ref", "subpop", "pref")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LLM_LOG_PATH = _PROJECT_ROOT / "logs" / "larc_nsga3_usage.jsonl"


def _normalized_objectives(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    span = np.maximum(f_max - f_min, 1e-12)
    return (F - f_min) / span


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0.0] / total
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(float(counts.size)))


def _associate_ref_dirs(F: np.ndarray, ref_dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F_norm = np.maximum(_normalized_objectives(F), 1e-12)
    R = np.maximum(np.asarray(ref_dirs, dtype=float), 1e-12)
    F_unit = F_norm / np.maximum(np.linalg.norm(F_norm, axis=1, keepdims=True), 1e-12)
    R_unit = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(F_unit @ R_unit.T, -1.0, 1.0)
    niche = np.argmax(cosine, axis=1)
    angle = np.arccos(cosine[np.arange(F.shape[0]), niche])
    return niche.astype(int), angle


class LARC_NSGA3(NSGA3Local):
    """NSGA-III with Two-Tier Bounded Semantic Controller."""

    ALGO_FLAGS = {"multi", "many", "real", "integer"}
    OBJECTIVE_SCOPE = "many"
    LOG_ALGORITHM_NAME = "LARC_NSGA3"
    DEFAULT_LOG_PATH = _DEFAULT_LLM_LOG_PATH

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: Any = None,
        sampling: Any = None,
        llm_client: Optional[LocalLLMClient] = None,
        llm_interval: int = 15,
        llm_log_path: Any = None,
        policy_temperature: float = 0.0,
        stagnation_window: int = 4,
        llm_max_share: float = 0.08,
        llm_min_gap: int = 10,
        action_reward_alpha: float = 0.25,
        max_action_streak: int = 5,
        min_llm_confidence: float = 0.0,
        n_max_evals_hint: int | None = None,
        llm_min_calls_abs: int = 6,
        llm_max_calls_abs: int = 15,
        enable_traps: bool = True,
        enable_llm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(pop_size=pop_size, ref_dirs=ref_dirs, sampling=sampling, **kwargs)
        self.enable_llm = bool(enable_llm)
        self.enable_traps = bool(enable_traps)

        if self.enable_llm:
            if llm_client is not None:
                self.llm_client = llm_client
            else:
                self.llm_client = LocalLLMClient(temperature=policy_temperature, max_tokens=48)
        else:
            self.llm_client = None

        if llm_log_path is None and self.enable_llm:
            llm_log_path = os.environ.get("EMOPYLAB_LARC_NSGA3_LOG")
        self.llm_log_path = None if llm_log_path in (None, False) else Path(llm_log_path)

        self.llm_interval = int(max(1, llm_interval))
        self.policy_temperature = float(max(0.0, min(policy_temperature, 1.0)))
        self.stagnation_window = int(max(2, stagnation_window))
        self.llm_max_share = float(max(0.05, min(llm_max_share, 0.95)))
        self.llm_min_gap = int(max(1, llm_min_gap))
        self.action_reward_alpha = float(max(0.05, min(action_reward_alpha, 1.0)))
        self.max_action_streak = int(max(2, max_action_streak))
        self.min_llm_confidence = float(max(0.0, min(min_llm_confidence, 1.0)))
        self._n_max_evals_hint = int(n_max_evals_hint) if n_max_evals_hint is not None else 0
        if self._n_max_evals_hint < 0:
            self._n_max_evals_hint = 0
        self.llm_min_calls_abs = int(max(1, llm_min_calls_abs))
        self.llm_max_calls_abs = int(max(self.llm_min_calls_abs, llm_max_calls_abs))
        self.current_action = "conv"
        self.action_history: list[dict[str, Any]] = []
        self.state_history: list[dict[str, Any]] = []
        self._best_norm_sum_history: list[float] = []
        self._action_reward_ema = {action: 0.0 for action in _ACTIONS}
        self._action_counts = {action: 0 for action in _ACTIONS}
        self._llm_queries = 0
        self._last_llm_generation = -10**9
        self._action_streak = 0
        self._state_action_outcomes: list[dict[str, Any]] = []
        self._max_history_shots: int = 3
        self._last_llm_state: dict[str, Any] = {}

        # Statistics for audit & rebuttal transparency
        self.llm_call_stats = {
            "total_queries": 0,
            "accepted_actions": 0,
            "rejected_by_trap": 0,
            "guardrail_overrides": 0,
            "graceful_fallbacks": 0,
            "tier1_heuristic_count": 0,
        }

    def _initialize_advance(self, infills=None, **kwargs: Any) -> None:
        super()._initialize_advance(infills=infills, **kwargs)
        if self.pop is not None and len(self.pop) > 0:
            state = self._population_state(self.pop)
            self._warm_start_ema(state)
            allowed = tuple(_ACTIONS)
            initial_action = self._heuristic_action(state, allowed)
            self.current_action = initial_action
            self._action_streak = 1
            record = {
                "generation": 0,
                "action": initial_action,
                "confidence": 1.0,
                "reason_code": "initial_heuristic",
                "source": "heuristic",
            }
            self.action_history.append(record)
            self.state_history.append(state)

    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return super()._infill()

        rng = rng_from_algo(self)
        cv = _constraint_violation(self.pop)
        fitness = self._selection_fitness(self.pop, self.current_action)
        if self.current_action == "var":
            mating = rng.integers(0, len(self.pop), size=self.pop_size)
        else:
            mating = np.asarray(TournamentSelection(2, self.pop_size, cv, fitness, rng=rng), dtype=int) - 1
            mating = np.clip(mating, 0, len(self.pop) - 1)

        params = self._variation_parameters(self.current_action)
        offspring = OperatorGA(self.problem, self.pop[mating], Parameter=params, rng=rng)
        if self.current_action == "var":
            offspring = self._variable_classification_mutation(offspring, rng)
        return offspring

    def _advance(self, infills=None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            return
        merged = Population.merge(self.pop, infills) if self.pop is not None and len(self.pop) else infills
        self.zmin = _update_zmin(self.zmin, merged, int(self.problem.n_obj))
        self._update_policy(merged)
        if self.current_action == "ref":
            self._adapt_reference_directions(merged)
        selected = _environmental_selection(
            merged,
            self.pop_size,
            np.asarray(self.ref_dirs, dtype=float),
            np.asarray(self.zmin, dtype=float),
            rng_from_algo(self),
        )
        if self.current_action == "subpop":
            selected = self._inject_subpopulation_diversity(selected, merged)
        self.pop = selected

    def _generation_index(self) -> int:
        return int(getattr(self, "n_gen", 0) or len(self.action_history))

    def _update_policy(self, pop: Population) -> None:
        state = self._population_state(pop)
        if self.action_history and self.state_history:
            previous = self.action_history[-1]
            self._credit_action(previous, self.state_history[-1], state)
        decision = self._select_policy_action(state)
        action = str(decision.get("action", "conv")).lower()
        if action not in _ACTIONS:
            action = "conv"
        if action == self.current_action:
            self._action_streak += 1
        else:
            self._action_streak = 1
        self.current_action = action
        record = {
            "generation": self._generation_index(),
            "action": self.current_action,
            "confidence": float(decision.get("confidence", 0.0)),
            "reason_code": str(decision.get("reason_code", "unknown")),
            "source": str(decision.get("source", "llm")),
        }
        self.action_history.append(record)
        self.state_history.append(state)

    def _population_state(self, pop: Population) -> dict[str, Any]:
        F = _population_objectives(pop)
        G = _population_constraints(pop)
        front_no, _ = NDSort(F, G, len(pop))
        front_no = np.asarray(front_no, dtype=float).reshape(-1)
        niche, angle = _associate_ref_dirs(F, np.asarray(self.ref_dirs, dtype=float))
        counts = np.bincount(niche, minlength=int(np.asarray(self.ref_dirs).shape[0]))
        entropy = _entropy_from_counts(counts)
        norm = _normalized_objectives(F)
        best_norm_sum = float(np.min(np.sum(norm, axis=1)))
        self._best_norm_sum_history.append(best_norm_sum)
        recent = self._best_norm_sum_history[-self.stagnation_window :]
        stagnation = 0 if len(recent) < self.stagnation_window else int(max(recent[:-1]) - recent[-1] <= 1e-4)
        prev_best = self._best_norm_sum_history[-2] if len(self._best_norm_sum_history) > 1 else best_norm_sum
        improvement = float(prev_best - best_norm_sum)

        evaluator = getattr(self, "evaluator", None)
        n_eval = int(getattr(evaluator, "n_eval", 0) or 0)
        n_max = self._resolve_n_max_evals()
        budget_ratio = 1.0 if n_max <= 0 else max(0.0, min(1.0, 1.0 - n_eval / float(n_max)))
        nd_ratio = float(np.mean(front_no == 1.0))
        angle_dispersion = float(np.mean(angle) / (np.pi / 2.0)) if angle.size else 0.0
        n_ref = int(np.asarray(self.ref_dirs).shape[0])
        empty_niches = float(np.sum(counts == 0)) / max(float(n_ref), 1.0)
        previous_state = self.state_history[-1] if self.state_history else None
        entropy_delta = (
            0.0 if previous_state is None else float(entropy - float(previous_state.get("crowding_entropy", entropy)))
        )
        nd_ratio_delta = (
            0.0
            if previous_state is None
            else float(nd_ratio - float(previous_state.get("non_dominated_ratio", nd_ratio)))
        )
        angle_dispersion_delta = (
            0.0
            if previous_state is None
            else float(angle_dispersion - float(previous_state.get("angle_dispersion", angle_dispersion)))
        )
        if budget_ratio > 0.66:
            phase = "early"
        elif budget_ratio > 0.33:
            phase = "middle"
        else:
            phase = "late"

        return {
            "n_obj": int(F.shape[1]),
            "generation": self._generation_index(),
            "budget_ratio": budget_ratio,
            "phase": phase,
            "crowding_entropy": entropy,
            "non_dominated_ratio": nd_ratio,
            "angle_dispersion": angle_dispersion,
            "angle_dispersion_delta": angle_dispersion_delta,
            "empty_niches": empty_niches,
            "best_norm_sum": best_norm_sum,
            "improvement": improvement,
            "entropy_delta": entropy_delta,
            "non_dominated_delta": nd_ratio_delta,
            "stagnation": stagnation,
            "allowed_actions": list(_ACTIONS),
        }

    def _select_policy_action(self, state: dict[str, Any]) -> dict[str, Any]:
        allowed = tuple(a for a in self._allowed_actions(state) if a in _ACTIONS)
        if not allowed:
            allowed = _ACTIONS

        if self.enable_llm and self._should_query_llm(state, allowed):
            decision = self._query_llm_policy(state, allowed)
            return self._guardrail_decision(decision, state, allowed)

        self.llm_call_stats["tier1_heuristic_count"] += 1
        heuristic_action = self._heuristic_action(state, allowed)
        return {
            "action": heuristic_action,
            "confidence": 0.75,
            "reason_code": "heuristic_between_llm",
            "source": "heuristic",
        }

    def _query_llm_policy(self, state: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
        self._llm_queries += 1
        self.llm_call_stats["total_queries"] += 1
        self._last_llm_generation = int(state.get("generation", self._generation_index()))
        narrative = self._state_to_narrative(state)
        history_shots = self._state_action_outcomes[-self._max_history_shots:] if self._state_action_outcomes else []

        prompt_payload = {
            "task": "select_nsga3_many_objective_policy_action",
            "state_narrative": narrative,
            "state": {k: state[k] for k in state if k != "allowed_actions"},
            "action_scores": {k: float(self._action_reward_ema.get(k, 0.0)) for k in allowed},
            "history": history_shots,
            "allowed_actions": list(allowed),
            "output_schema": {"action": "string", "confidence": "float", "reason_code": "string"},
        }
        self._last_llm_state = dict(state)

        system_prompt = (
            "You are the discrete controller of LARC-NSGA3 for many-objective optimization. "
            "You must select exactly one action from allowed_actions. "
            "Output ONLY a valid JSON object matching: "
            '{"action": "<chosen_action>", "confidence": <0.0-1.0>, "reason_code": "<short_reason>"}'
        )

        try:
            if self.llm_client is None:
                raise RuntimeError("LLM client not configured.")

            # Dispatch to native LocalLLMClient (handles mlx-lm or in-process llama-cpp)
            data = self.llm_client.json_call(
                prompt=json.dumps(prompt_payload, sort_keys=True),
                system=system_prompt,
            )

            if not isinstance(data, dict) or "action" not in data:
                raise ValueError("Malformed LLM response JSON.")

            action = str(data.get("action", "")).strip().lower()
            confidence = max(0.0, min(float(data.get("confidence", 0.5)), 1.0))
            reason_code = str(data.get("reason_code", "llm_policy"))

            if confidence < self.min_llm_confidence:
                raise ValueError(f"Confidence {confidence:.2f} below threshold {self.min_llm_confidence:.2f}.")

            if action not in allowed:
                # LLM suggested action outside current safe allowed set
                self.llm_call_stats["rejected_by_trap"] += 1
                fallback_action = max(allowed, key=lambda a: float(self._action_reward_ema.get(a, 0.0)))
                decision = {
                    "action": fallback_action,
                    "confidence": 0.3,
                    "reason_code": "trap_filtered_llm_fallback",
                    "source": "heuristic",
                    "_llm_suggested": action,
                }
                self._write_llm_usage_log(decision, state)
                return decision

            self.llm_call_stats["accepted_actions"] += 1
            decision = {
                "action": action,
                "confidence": confidence,
                "reason_code": reason_code,
                "source": "llm",
            }
            self._write_llm_usage_log(decision, state)
            return decision

        except Exception as exc:
            # Graceful Fallback: Never crash or terminate the run.
            self.llm_call_stats["graceful_fallbacks"] += 1
            fallback_action = self._heuristic_action(state, allowed)
            decision = {
                "action": fallback_action,
                "confidence": 0.5,
                "reason_code": f"graceful_fallback_error: {type(exc).__name__}",
                "source": "heuristic",
            }
            self._write_llm_usage_log(decision, state)
            return decision

    def _write_llm_usage_log(self, decision: dict[str, Any], state: dict[str, Any]) -> None:
        if self.llm_log_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm": self.LOG_ALGORITHM_NAME,
            "problem_name": self._problem_name(),
            "model": str(getattr(self.llm_client, "resolved_model", "none")),
            "generation": int(state.get("generation", self._generation_index())),
            "action": str(decision.get("action", "")),
            "confidence": float(decision.get("confidence", 0.0)),
            "reason_code": str(decision.get("reason_code", "")),
            "source": str(decision.get("source", "llm")),
            "llm_interval": self.llm_interval,
            "llm_query_count": int(self._llm_queries),
        }
        try:
            self.llm_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.llm_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            pass

    def _allowed_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if not self.enable_traps:
            return _ACTIONS

        stagnation = int(state.get("stagnation", 0)) > 0
        empty_niches = float(state.get("empty_niches", 0.0))
        nd_ratio = float(state.get("non_dominated_ratio", 0.0))

        # Degenerate/Disconnected front trap
        if stagnation and empty_niches > 0.40 and nd_ratio > 0.50:
            return ("pref", "conv")

        # Multimodal trap
        if stagnation and nd_ratio < 0.20:
            return ("pref", "conv", "ref")

        phase = str(state.get("phase", "middle"))
        if phase == "early":
            return ("div", "subpop", "ref", "var", "conv")
        if phase == "late":
            return ("conv", "pref", "ref", "subpop")
        return _ACTIONS

    def _llm_call_budget(self) -> int:
        n_max = self._resolve_n_max_evals()
        if n_max <= 0:
            n_max = max(1, int(max(100, self.pop_size * 100)))
        est_generations = int(max(1, np.ceil(n_max / float(max(1, self.pop_size)))))
        share_budget = int(max(2, np.ceil(self.llm_max_share * est_generations)))
        return int(min(self.llm_max_calls_abs, max(self.llm_min_calls_abs, share_budget)))

    def _resolve_n_max_evals(self) -> int:
        if self._n_max_evals_hint > 0:
            return int(self._n_max_evals_hint)
        term = getattr(self, "termination", None)
        if term is not None:
            for attr in ("n_max_evals", "n_max_eval", "max_evals"):
                value = getattr(term, attr, None)
                if value is None:
                    continue
                try:
                    n_max = int(value)
                    if n_max > 0:
                        return n_max
                except Exception:
                    continue
        return 0

    def _should_query_llm(self, state: dict[str, Any], allowed: tuple[str, ...]) -> bool:
        if not self.enable_llm or not allowed:
            return False
        generation = int(state.get("generation", 0))
        if generation - self._last_llm_generation < self.llm_min_gap:
            return False
        if self._llm_queries >= self._llm_call_budget():
            return False

        periodic = generation % self.llm_interval == 0
        stagnation = int(state.get("stagnation", 0)) > 0
        low_diversity = float(state.get("crowding_entropy", 0.0)) < 0.45
        low_nd = float(state.get("non_dominated_ratio", 0.0)) < 0.20
        in_budget = float(state.get("budget_ratio", 1.0)) > 0.10
        state_drifted = in_budget and self._state_drift(state) > 0.15
        return bool(periodic or (in_budget and stagnation) or (in_budget and low_diversity and low_nd) or state_drifted)

    def _heuristic_action(self, state: dict[str, Any], allowed: tuple[str, ...]) -> str:
        stagnation = int(state.get("stagnation", 0)) > 0
        entropy = float(state.get("crowding_entropy", 0.0))
        empty_niches = float(state.get("empty_niches", 0.0))
        nd_ratio = float(state.get("non_dominated_ratio", 0.0))
        improvement = float(state.get("improvement", 0.0))

        # 1. Extreme fail-safes: Population collapsed
        if entropy < 0.20 and "div" in allowed:
            return "div"

        # 2. Progress inertia: Do not interrupt working convergence
        if improvement > 1e-4 and "conv" in allowed:
            return "conv"

        # 3. Degenerate/Disconnected front trap
        if stagnation and empty_niches > 0.40 and nd_ratio > 0.50:
            safe_actions = [a for a in ("pref", "conv") if a in allowed]
            if safe_actions:
                scores = {a: float(self._action_reward_ema.get(a, 0.0)) for a in safe_actions}
                return max(scores, key=scores.get) if scores else safe_actions[0]

        # 4. Multimodal trap
        if stagnation and nd_ratio < 0.20:
            if "pref" in allowed:
                return "pref"
            if "conv" in allowed:
                return "conv"

        # 5. Local Reinforcement Learning (EMA-guided)
        scores = {a: float(self._action_reward_ema.get(a, 0.0)) for a in allowed}
        best = max(scores, key=scores.get) if scores else "conv"

        # Conservation bias: only switch from 'conv' if advantage is clear
        if best != "conv" and "conv" in allowed:
            if scores[best] < scores.get("conv", 0.0) + 0.05:
                return "conv"

        return best

    def _credit_action(self, previous: dict[str, Any], _prev_state: dict[str, Any], cur_state: dict[str, Any]) -> None:
        action = str(previous.get("action", "conv")).lower()
        if action not in _ACTIONS:
            return
        improvement = float(cur_state.get("improvement", 0.0))
        nd_delta = float(cur_state.get("non_dominated_delta", 0.0))
        entropy_delta = float(cur_state.get("entropy_delta", 0.0))
        entropy = float(cur_state.get("crowding_entropy", 0.5))
        empty_niches = float(cur_state.get("empty_niches", 0.0))
        stagnation = int(cur_state.get("stagnation", 0))
        angle_disp_delta = float(cur_state.get("angle_dispersion_delta", 0.0))

        angle_term = np.tanh(-8.0 * angle_disp_delta)
        improvement_term = np.tanh(30.0 * improvement)
        nd_term = np.tanh(6.0 * nd_delta)
        entropy_term = np.tanh(6.0 * entropy_delta)

        if action == "conv":
            reward = 0.55 * improvement_term + 0.25 * nd_term + 0.10 * entropy_term + 0.10 * angle_term
        elif action == "div":
            reward = 0.15 * improvement_term + 0.10 * nd_term + 0.60 * entropy_term + 0.15 * angle_term
        elif action == "ref":
            reward = 0.35 * improvement_term + 0.15 * nd_term + 0.20 * entropy_term + 0.30 * angle_term
        elif action == "subpop":
            reward = 0.25 * improvement_term + 0.15 * nd_term + 0.35 * entropy_term + 0.25 * angle_term
        elif action == "pref":
            reward = 0.50 * improvement_term + 0.30 * nd_term + 0.10 * entropy_term + 0.10 * angle_term
        else:  # "var"
            reward = 0.30 * improvement_term + 0.15 * nd_term + 0.40 * entropy_term + 0.15 * angle_term

        if stagnation and improvement <= 1e-5:
            if entropy < 0.30:
                reward -= 0.30
            elif empty_niches > 0.40:
                reward -= 0.25
            else:
                reward -= 0.10

        if action == "div" and entropy_delta > 0.05:
            reward += 0.15

        outcome_record = {
            "gen": int(cur_state.get("generation", 0)),
            "action": action,
            "improvement": round(improvement, 5),
            "entropy_delta": round(entropy_delta, 3),
            "nd_delta": round(nd_delta, 3),
        }
        self._state_action_outcomes.append(outcome_record)
        if len(self._state_action_outcomes) > 20:
            self._state_action_outcomes = self._state_action_outcomes[-20:]

        previous_reward = float(self._action_reward_ema.get(action, 0.0))
        alpha = self.action_reward_alpha
        self._action_reward_ema[action] = (1.0 - alpha) * previous_reward + alpha * float(reward)
        self._action_counts[action] = int(self._action_counts.get(action, 0)) + 1

    def _guardrail_decision(
        self,
        decision: dict[str, Any],
        state: dict[str, Any],
        allowed: tuple[str, ...],
    ) -> dict[str, Any]:
        action = str(decision.get("action", "conv")).lower()
        if action not in allowed:
            action = max(allowed, key=lambda a: float(self._action_reward_ema.get(a, 0.0)))
            decision["action"] = action
            decision["reason_code"] = "guardrail_clamped_action"

        if action == self.current_action and self._action_streak >= self.max_action_streak:
            current_reward = float(self._action_reward_ema.get(action, 0.0))
            alternatives = [item for item in allowed if item != action]
            if alternatives:
                best_alt = max(alternatives, key=lambda item: float(self._action_reward_ema.get(item, -1e9)))
                best_reward = float(self._action_reward_ema.get(best_alt, -1e9))
                budget_ratio = float(state.get("budget_ratio", 0.5))
                adaptive_threshold = max(0.01, 0.05 * (1.0 - budget_ratio))
                if best_reward >= current_reward + adaptive_threshold:
                    self.llm_call_stats["guardrail_overrides"] += 1
                    decision = {
                        "action": best_alt,
                        "confidence": float(decision.get("confidence", 0.5)),
                        "reason_code": "anti_collapse_guardrail",
                        "source": "guardrail",
                    }
        return decision

    def _state_to_narrative(self, state: dict[str, Any]) -> str:
        H = float(state.get("crowding_entropy", 0.5))
        nd = float(state.get("non_dominated_ratio", 0.5))
        eps = float(state.get("empty_niches", 0.0))
        imp = float(state.get("improvement", 0.0))
        stag = int(state.get("stagnation", 0)) > 0
        phase = str(state.get("phase", "middle"))
        budget = float(state.get("budget_ratio", 0.5))
        M = int(state.get("n_obj", 5))
        gen = int(state.get("generation", 0))
        ad = float(state.get("angle_dispersion", 0.0))

        parts: list[str] = [f"Gen {gen} | Phase: {phase} ({budget:.0%} budget left) | M={M} objectives."]

        if H < 0.20:
            parts.append(f"CRITICAL: Population collapsed (entropy={H:.2f}). Diversity injection is urgent.")
        elif H < 0.40:
            parts.append(f"Low diversity (entropy={H:.2f}). Niche coverage weakening.")
        elif H > 0.75:
            parts.append(f"High diversity (entropy={H:.2f}). Good coverage; convergence pressure is beneficial.")
        else:
            parts.append(f"Moderate diversity (entropy={H:.2f}).")

        if stag:
            if imp <= 1e-5:
                parts.append("STAGNATION: No measurable objective improvement.")
            else:
                parts.append(f"Near-stagnation: improvement={imp:.5f}.")
        elif imp > 1e-3:
            parts.append(f"Active convergence (improvement={imp:.4f}).")

        if eps > 0.40 and nd > 0.50:
            parts.append(f"Degenerate/disconnected front: {eps:.0%} niches empty, {nd:.0%} non-dominated. Polish front.")
        elif nd < 0.20:
            parts.append(f"Multimodal trap: low non-dominated ratio ({nd:.0%}). Strengthen selection pressure.")
        elif eps > 0.25:
            parts.append(f"Partial niche coverage: {eps:.0%} niches empty.")

        if ad > 0.60:
            parts.append(f"Wide angular spread ({ad:.2f}).")
        elif ad < 0.15:
            parts.append(f"Tight angular alignment ({ad:.2f}).")

        return " ".join(parts)

    def _state_drift(self, current: dict[str, Any]) -> float:
        if not self._last_llm_state:
            return 0.0
        keys = ["crowding_entropy", "non_dominated_ratio", "empty_niches", "improvement"]
        drift = sum(
            abs(float(current.get(k, 0.0)) - float(self._last_llm_state.get(k, 0.0)))
            for k in keys
        )
        return float(drift)

    def _warm_start_ema(self, state: dict[str, Any]) -> None:
        eps = float(state.get("empty_niches", 0.0))
        M = int(state.get("n_obj", 5))
        nd = float(state.get("non_dominated_ratio", 0.5))

        self._action_reward_ema["conv"] = 0.10
        if eps > 0.30:
            self._action_reward_ema["pref"] = 0.15
            self._action_reward_ema["ref"] = 0.10
        if M >= 8:
            self._action_reward_ema["subpop"] = 0.12
            self._action_reward_ema["div"] = 0.08
        if nd > 0.60:
            self._action_reward_ema["pref"] = max(self._action_reward_ema.get("pref", 0.0), 0.12)
            self._action_reward_ema["conv"] = 0.14

    def _problem_name(self) -> str:
        problem = getattr(self, "problem", None)
        if problem is None:
            return "unknown"
        name = getattr(problem, "name", None)
        if callable(name):
            try:
                val = name()
                if val:
                    return str(val)
            except Exception:
                pass
        if isinstance(name, str) and name:
            return name
        return problem.__class__.__name__

    def _selection_fitness(self, pop: Population, action: str) -> np.ndarray:
        # Strict zero-pressure random mating (avoids destroying boundary diversity in MaOP)
        return np.zeros(len(pop), dtype=float)

    def _variation_parameters(self, action: str) -> list[float]:
        n_obj = int(getattr(getattr(self, "problem", None), "n_obj", 5) or 5)
        eta_c_base = max(7.0, 20.0 - max(0, n_obj - 5) * 1.5)
        eta_m_base = max(15.0, 20.0 - max(0, n_obj - 5) * 0.5)
        if action == "conv":
            return [1.0, eta_c_base, 1.0, eta_m_base]
        if action == "div":
            eta_c_div = max(eta_c_base - 4.0, 7.0)
            return [1.0, eta_c_div, 1.5, max(eta_m_base - 3.0, 12.0)]
        if action == "var":
            return [1.0, eta_c_base, 3.0, max(eta_m_base - 5.0, 10.0)]
        if action == "subpop":
            return [0.9, eta_c_base, 1.0, eta_m_base]
        if action == "pref":
            return [1.0, 35.0, 0.5, 35.0]
        return [1.0, eta_c_base, 1.0, eta_m_base]

    def _adapt_reference_directions(self, pop: Population) -> None:
        F = _population_objectives(pop)
        norm = _normalized_objectives(F)
        niche, _ = _associate_ref_dirs(F, np.asarray(self.ref_dirs, dtype=float))
        counts = np.bincount(niche, minlength=int(np.asarray(self.ref_dirs).shape[0]))
        active = np.where(counts > 0)[0]
        if active.size == 0:
            return
        centroid = np.mean(norm, axis=0)
        centroid = centroid / max(float(np.sum(centroid)), 1e-12)
        ref_dirs = np.asarray(self.ref_dirs, dtype=float)
        inactive = np.where(counts == 0)[0]
        if inactive.size:
            ref_dirs[inactive] = 0.90 * ref_dirs[inactive] + 0.10 * centroid[None, :]
        ref_dirs[active] = 0.98 * ref_dirs[active] + 0.02 * centroid[None, :]
        ref_dirs = ref_dirs / np.maximum(np.sum(ref_dirs, axis=1, keepdims=True), 1e-12)
        self.ref_dirs = np.maximum(ref_dirs, 1e-12)

    def _variable_classification_mutation(self, offspring: Population, rng: np.random.Generator) -> Population:
        X = np.asarray(offspring.get("X"), dtype=float)
        if X.size == 0:
            return offspring
        xl = np.asarray(self.problem.xl, dtype=float).reshape(1, -1)
        xu = np.asarray(self.problem.xu, dtype=float).reshape(1, -1)
        span = np.maximum(xu - xl, 1e-12)
        n_var = X.shape[1]
        n_mut = max(1, int(np.ceil(0.15 * n_var)))
        cols = rng.choice(n_var, size=n_mut, replace=False)
        X_new = X.copy()
        X_new[:, cols] = np.clip(X_new[:, cols] + rng.normal(0.0, 0.08, size=(X.shape[0], n_mut)) * span[:, cols], xl[:, cols], xu[:, cols])
        return Population.new("X", X_new)

    def _inject_subpopulation_diversity(self, selected: Population, merged: Population) -> Population:
        if len(selected) >= self.pop_size or len(merged) <= len(selected):
            return selected
        need = int(self.pop_size - len(selected))
        F_merged = _population_objectives(merged)
        ref_dirs = np.asarray(self.ref_dirs, dtype=float)
        niche_merged, angle_merged = _associate_ref_dirs(F_merged, ref_dirs)
        n_ref = int(ref_dirs.shape[0])
        F_selected = _population_objectives(selected)
        niche_selected, _ = _associate_ref_dirs(F_selected, ref_dirs)
        covered = set(niche_selected.tolist())
        empty_niches = [r for r in range(n_ref) if r not in covered]
        injected_indices: list[int] = []
        for ref_idx in empty_niches:
            if len(injected_indices) >= need:
                break
            candidates = np.where(niche_merged == ref_idx)[0]
            if candidates.size == 0:
                continue
            best_local = int(candidates[int(np.argmin(angle_merged[candidates]))])
            if best_local not in injected_indices:
                injected_indices.append(best_local)
        if len(injected_indices) < need:
            selected_set = set(injected_indices)
            order = np.argsort(angle_merged, kind="mergesort")
            for idx in order:
                if len(injected_indices) >= need:
                    break
                if int(idx) not in selected_set:
                    injected_indices.append(int(idx))
        if not injected_indices:
            return selected
        extra = merged[np.array(injected_indices, dtype=int)]
        return Population.merge(selected, extra)
