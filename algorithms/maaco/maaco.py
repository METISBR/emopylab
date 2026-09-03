# -*- coding: utf-8 -*-
"""MaACO v2 — Many-Objective Ant Colony Optimization with
LLM-as-Operator and Contextual-Bandit Operator Selection.

Authors
-------
Thiago Santos, Sebastiao Xavier (UFOP / METISBr).

Paper
-----
"MaACO: An Angle-Penalized Ant Colony Optimization Algorithm with
LLM-Guided Adaptation for Many-Objective Problems" (ACDSA 2027).

Plugin
------
This file is consumed by the EmoPyLab plugin system.  The class
``MaACO`` exposes the standard EmoPyLab ``Algorithm`` interface and is
swappable with ``NSGA3`` / ``RVEA`` / ``MOEAD`` in any benchmark
script.

Architecture (v2)
-----------------
MaACO combines four symbiotic components:

1. **Operator Library** (``algorithms.maaco.operators``):
   - ``llm_sbx`` (Simulated Binary Crossover with dynamic ``eta_c``)
   - ``llm_de`` (Differential Evolution with dynamic ``F, CR``)
   - ``llm_perturb`` (Polynomial Mutation with dynamic ``eta_m``)
   - ``acor_mixture`` (Classical ACOR Gaussian-mixture kernel sampling)
2. **Contextual Multi-Armed Bandit** (``algorithms.maaco.bandit.UCB1OperatorBandit``):
   Dynamically selects the operator at each generation based on UCB1
   scores over recent hypervolume improvements.
3. **LLM Meta-Controller** (``core.llm.LocalLLMClient``):
   Every ``llm_period`` generations, queries the local
   ``Qwen2.5-0.5B-Instruct-4bit`` model served via ``mlx-lm`` on
   Apple Silicon (or ``llama-cpp-python`` in-process on Windows/Linux).
   The LLM proposes:
     - An operator preference (``operator_choice``)
     - Fine-grained operator parameters (``operator_params``)
     - Classical hyperparameter adjustments (``q, xi, rho, apd_alpha``)
4. **Angle-Penalized Environmental Selection (APD)**:
   RVEA-style selection with dimension-safe reference vectors that
   preserves both convergence and diversity across irregular fronts.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from operators.sampling.lhs import LHS
from operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.ref_dirs import get_reference_directions

from core.llm import (
    DEFAULT_BASE_URL as _DEFAULT_LLM_BASE_URL,
    DEFAULT_MODEL as _DEFAULT_LLM_MODEL,
    LocalLLMClient,
)

from .bandit import UCB1OperatorBandit
from .operators import OPERATOR_NAMES, apply_operator


logger = logging.getLogger(__name__)


class LLMPolicyError(RuntimeError):
    """Raised when a required MaACO local-policy call cannot be applied.

    ``status`` and ``event`` are positional-default fields so the exception is
    serializable through ``ProcessPoolExecutor`` on macOS and Windows.
    """

    def __init__(self, message: str, status: str = "unknown",
                 event: Optional[Dict[str, Any]] = None):
        super().__init__(message, status, event or {})
        self.status = status
        self.event = dict(event or {})


# ---------------------------------------------------------------------------
# LLM Prompt Template (extended for operator selection + bandit feedback)
# ---------------------------------------------------------------------------

_LLM_PROMPT_TEMPLATE = """Given multi-objective optimization state:
- Generation: {generation}, Objectives: {n_obj}
- Archive front size: {front_size}/{ref_count}
- Stagnation count: {stagnation}
- Operator reward history: {bandit_means}

You must output ONLY valid JSON matching this exact structure:
{{"operator_choice": "llm_de", "operator_params": {{"F": 0.8, "CR": 0.9, "eta_c": 20.0, "eta_m": 20.0, "q": 1.0, "xi": 0.85}}, "apd_alpha": 2.0}}

Replace "llm_de" with one of ["llm_sbx", "llm_de", "llm_perturb", "acor_mixture"]:
- If stagnation > 0: use "llm_de" with F=0.8, CR=0.9
- If front_size < {ref_count}/2: use "acor_mixture" with q=1.2
- Else: use "llm_sbx" (eta_c=20.0) or "llm_perturb" (eta_m=20.0)
"""


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------

def _call_llm(stats: Dict[str, Any],
              client: Optional[LocalLLMClient] = None) -> Optional[Dict[str, Any]]:
    """Dispatch the state to the LLM and return the parsed JSON proposal."""
    if client is None:
        client = LocalLLMClient()
    prompt = _LLM_PROMPT_TEMPLATE.format(
        generation=stats.get("generation", 0),
        n_obj=stats.get("n_obj", 2),
        pop_size=stats.get("pop_size", 100),
        front_size=stats.get("front_size", 0),
        ref_count=stats.get("ref_count", 0),
        stagnation=stats.get("stagnation", 0),
        bandit_means=json.dumps(stats.get("bandit_means", {})),
    )
    proposal = client.json_call(
        prompt,
        system="You are an adaptive operator selection policy. Output ONLY JSON.",
    )
    if isinstance(proposal, dict):
        return proposal
    return None


# ---------------------------------------------------------------------------
# ALGORITHM_FLAGS for EmoPyLab plugin discovery
# ---------------------------------------------------------------------------

ALGORITHM_FLAGS = {
    "MaACO": {"multi", "many"},
}


# ---------------------------------------------------------------------------
# Public algorithm class
# ---------------------------------------------------------------------------

class MaACO(Algorithm):
    """Many-Objective Ant Colony Optimization (v2 — LLM + Bandit Driven).

    Parameters
    ----------
    pop_size : int, default=100
        Population / archive size.
    ref_dirs : np.ndarray, optional
        Reference directions (K, M).
    ref_partitions : int, default=12
        Das-Dennis partition count when ref_dirs is None.
    llm_base_url : str
        OpenAI-compatible endpoint.  Defaults to ``http://localhost:8080/v1``.
    llm_model : str
        Model name.  Defaults to ``mlx-community/Qwen2.5-0.5B-Instruct-4bit``.
    use_llm : bool, default=True
        Enables the periodic language-model policy supervisor. Set False for
        the MaACO-NoLLM causal ablation; the UCB1 operator selector and APD
        survival remain active.
    llm_period : int, default=25
        Generation interval between LLM calls when ``use_llm`` is True.
    adapt_period : int, default=50
        Generation interval between reference vector adaptations.
    bandit_c : float, default=0.5
        UCB1 exploration constant for operator selection.
    seed : int, optional
        Random seed.
    """

    def __init__(self,
                 pop_size: int = 100,
                 ref_dirs: Optional[np.ndarray] = None,
                 ref_partitions: int = 12,
                 use_llm: bool = True,
                 llm_base_url: str = _DEFAULT_LLM_BASE_URL,
                 llm_model: str = _DEFAULT_LLM_MODEL,
                 llm_client: Optional[LocalLLMClient] = None,
                 llm_event_path: Optional[str | Path] = None,
                 llm_period: int = 25,
                 adapt_period: int = 50,
                 bandit_c: float = 0.5,
                 seed: Optional[int] = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if ref_dirs is not None:
            self.ref_dirs = np.asarray(ref_dirs, dtype=float)
        else:
            self._ref_partitions = ref_partitions
            self.ref_dirs = None

        self.pop_size = int(pop_size)

        # LLM setup. The MaACO-NoLLM ablation disables only this policy
        # supervisor; all evolutionary operators and the UCB1 bandit remain.
        self.use_llm = bool(use_llm)
        self.llm_base_url = str(llm_base_url)
        self.llm_model = str(llm_model)
        self.llm_period = max(1, int(llm_period))
        self.adapt_period = max(1, int(adapt_period))
        self._llm_client: Optional[LocalLLMClient] = llm_client
        self._llm_call_count = 0
        self._llm_applied_count = 0
        self._llm_failed_count = 0
        self._llm_no_response_count = 0
        self._llm_invalid_response_count = 0
        self._llm_schema_rejected_count = 0
        self._llm_noop_count = 0
        self._llm_events: List[Dict[str, Any]] = []
        self._llm_last_status = "disabled" if not self.use_llm else "not_called"
        self._llm_latency_ms: List[float] = []
        self._llm_event_path = Path(llm_event_path) if llm_event_path else None
        self._llm_event_sink: Optional[Callable[[Dict[str, Any]], None]] = None
        self._llm_preference: Optional[str] = None
        if self._llm_event_path is not None:
            self._llm_event_path.parent.mkdir(parents=True, exist_ok=True)
            def _jsonl_sink(event: Dict[str, Any]) -> None:
                with self._llm_event_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
            self._llm_event_sink = _jsonl_sink
        elif llm_client is not None and hasattr(llm_client, "json_call"):
            # Tests can provide a fake client; no filesystem side effect occurs.
            self._llm_event_sink = None
        self._llm_preference: Optional[str] = None
        self._operator_params: Dict[str, Any] = {
            "eta_c": 20.0,
            "prob_c": 1.0,
            "F": 0.5,
            "CR": 0.9,
            "eta_m": 20.0,
            "q": 1.0,
            "xi": 0.85,
        }

        # Multi-armed bandit for dynamic operator selection
        self.bandit = UCB1OperatorBandit(
            arm_names=OPERATOR_NAMES,
            c=bandit_c,
            window_size=20,
        )

        # Environmental selection parameters
        self.apd_alpha = 2.0

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._hv_history: List[float] = []
        self._last_hv: float = 0.0
        self._stagnation = 0
        self._last_operator_used: str = "llm_sbx"

    # ------------------------------------------------------------------
    # EmoPyLab.Algorithm interface
    # ------------------------------------------------------------------

    def _setup(self, problem, **kwargs):
        n_obj = int(problem.n_obj)
        if self.ref_dirs is None or self.ref_dirs.shape[1] != n_obj:
            self.ref_dirs = np.asarray(
                get_reference_directions(
                    "das-dennis", n_obj,
                    n_partitions=getattr(self, '_ref_partitions', 12)),
                dtype=float,
            )
        if self.pop_size < self.ref_dirs.shape[0]:
            self.pop_size = int(self.ref_dirs.shape[0])

    def _initialize_infill(self) -> Population:
        return LHS().do(self.problem, self.pop_size)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._t = 0
        F = np.asarray(self.pop.get("F"), dtype=float)
        self._last_hv = _fast_hv_estimate(F)

    def _infill(self) -> Population:
        """Generate offspring using the bandit-selected operator."""
        X_pop = np.asarray(self.pop.get("X"), dtype=float)
        F_pop = np.asarray(self.pop.get("F"), dtype=float)
        n_pop, n_var = X_pop.shape
        xl = np.asarray(self.problem.xl, dtype=float)
        xu = np.asarray(self.problem.xu, dtype=float)

        if n_pop == 0:
            return Population.new(
                "X", self._rng.uniform(xl, xu, size=(self.pop_size, n_var))
            )

        # 1. Bandit selects the operator (informed by LLM preference)
        chosen_op = self.bandit.select(
            self._rng,
            llm_preference=self._llm_preference,
        )
        self._last_operator_used = chosen_op

        # 2. Sort archive by NSGA-II non-dominated rank + crowding
        ranks, crowding = _nsga2_ranks(F_pop)
        order = np.lexsort((-crowding, ranks))
        X_sorted = X_pop[order]
        F_sorted = F_pop[order]

        # 3. Apply the chosen operator
        X_off = apply_operator(
            chosen_op,
            X_sorted,
            F_sorted,
            self.problem,
            self._rng,
            self.pop_size,
            params=self._operator_params,
        )

        return Population.new("X", X_off)

    def _advance(self, infills=None, **kwargs: Any) -> None:
        """Environmental selection (APD) + reward computation + LLM query."""
        if infills is not None:
            self.pop = Population.merge(self.pop, infills)

        F = np.asarray(self.pop.get("F"), dtype=float)
        if F.size == 0:
            return

        # 1. APD Environmental Selection
        survivors = self._apd_selection(F)
        self.pop = self.pop[survivors]
        F_surv = np.asarray(self.pop.get("F"), dtype=float)

        # 2. Compute hypervolume reward for the bandit
        current_hv = _fast_hv_estimate(F_surv)
        reward = max(0.0, current_hv - self._last_hv)
        self.bandit.update(self._last_operator_used, reward)
        self._last_hv = current_hv

        # Track stagnation
        self._hv_history.append(round(current_hv, 4))
        if len(self._hv_history) >= 4:
            recent = self._hv_history[-4:]
            if max(recent) - min(recent) < 1e-4:
                self._stagnation += 1
            else:
                self._stagnation = 0

        self._t += 1

        # 3. LLM-Guided Adaptation (disabled only for MaACO-NoLLM ablation)
        if self.use_llm and self._t % self.llm_period == 0:
            self._maybe_llm_call()

        # 4. Reference Vector Adaptation
        if self._t % self.adapt_period == 0:
            try:
                self.ref_dirs = self._adapt_reference_directions(
                    F_surv, self.ref_dirs)
            except Exception as exc:
                logger.debug("Ref adaptation skipped: %s", exc)

    # ------------------------------------------------------------------
    # LLM Interaction
    # ------------------------------------------------------------------

    def _record_llm_event(self, event: Dict[str, Any]) -> None:
        """Store one compact immutable policy event and notify an optional sink."""
        payload = dict(event)
        self._llm_events.append(payload)
        if self._llm_event_sink is not None:
            self._llm_event_sink(payload)

    def _fail_llm_policy(self, status: str, detail: str, event: Dict[str, Any]) -> None:
        """Record a required policy failure and abort the enabled-mode run."""
        self._llm_failed_count += 1
        self._llm_last_status = status
        self._record_llm_event(event)
        raise LLMPolicyError(
            f"Required MaACO LLM policy call failed: {status}: {detail}",
            status=status,
            event=event,
        )

    @staticmethod
    def _finite_value(value: Any, name: str) -> float:
        """Convert a scalar to a finite float or raise a schema validation error."""
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _validate_llm_proposal(self, raw: Any) -> tuple[str, Dict[str, Any], List[str], str]:
        """Validate a raw LLM policy proposal before any state mutation.

        Returns ``(status, normalized_proposal, changed_fields, detail)``.
        Status is either ``applied`` or ``noop``; invalid inputs raise
        ``ValueError`` so required mode can stop immediately.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("proposal is not a JSON object")

        normalized: Dict[str, Any] = {}
        changed: List[str] = []

        if "operator_choice" in raw:
            choice = str(raw["operator_choice"]).strip().lower()
            if choice not in OPERATOR_NAMES:
                raise ValueError(f"operator_choice={choice!r} is not recognized")
            normalized["operator_choice"] = choice
            if choice != self._llm_preference:
                changed.append("operator_choice")

        if "operator_params" in raw:
            params = raw["operator_params"]
            if not isinstance(params, Mapping):
                raise ValueError("operator_params is not a mapping")
            normalized_params: Dict[str, float] = {}
            ranges = {
                "eta_c": (2.0, 50.0),
                "F": (0.1, 1.2),
                "CR": (0.05, 1.0),
                "eta_m": (5.0, 60.0),
                "q": (0.05, 2.0),
                "xi": (0.1, 1.0),
            }
            for name, (lower, upper) in ranges.items():
                if name not in params:
                    continue
                value = self._finite_value(params[name], name)
                if value < lower or value > upper:
                    raise ValueError(f"{name}={value} outside [{lower}, {upper}]")
                normalized_params[name] = value
                if not math.isclose(value, float(self._operator_params[name]), rel_tol=0.0, abs_tol=1e-12):
                    changed.append(name)
            normalized["operator_params"] = normalized_params

        if "apd_alpha" in raw:
            alpha = self._finite_value(raw["apd_alpha"], "apd_alpha")
            if alpha < 0.5 or alpha > 5.0:
                raise ValueError(f"apd_alpha={alpha} outside [0.5, 5.0]")
            normalized["apd_alpha"] = alpha
            if not math.isclose(alpha, self.apd_alpha, rel_tol=0.0, abs_tol=1e-12):
                changed.append("apd_alpha")

        if not normalized:
            raise ValueError("proposal has no recognized policy fields")
        return ("applied" if changed else "noop"), normalized, changed, ""

    def _maybe_llm_call(self) -> None:
        if getattr(self, "problem", None) is None:
            return

        if self._llm_client is None:
            self._llm_client = LocalLLMClient(
                base_url=self.llm_base_url,
                model=self.llm_model,
            )

        F_cur = np.asarray(self.pop.get("F"), dtype=float)
        alloc_state = self.bandit.get_state()
        stats = {
            "generation": self._t,
            "n_obj": int(self.problem.n_obj),
            "pop_size": int(len(self.pop)),
            "front_size": int(F_cur.shape[0]),
            "ref_count": int(self.ref_dirs.shape[0]),
            "hv_history": self._hv_history[-5:],
            "stagnation": self._stagnation,
            "last_operator": self._last_operator_used,
            "bandit_means": alloc_state.get("recent_mean_rewards", {}),
            "bandit_counts": alloc_state.get("counts", {}),
        }
        before_preference = self._llm_preference
        before_params = dict(self._operator_params)
        before_alpha = self.apd_alpha
        self._llm_call_count += 1

        try:
            raw = _call_llm(stats, client=self._llm_client)
        except Exception as exc:
            status_info = getattr(self._llm_client, "last_call_status", {})
            event = {
                "generation": self._t,
                "status": "transport_error",
                "detail": str(exc),
                "operator_before": before_preference,
                "operator_after": before_preference,
                "changed": [],
                **dict(status_info),
            }
            self._fail_llm_policy("transport_error", str(exc), event)
            return

        status_info = dict(getattr(self._llm_client, "last_call_status", {}))
        if raw is None:
            status = str(status_info.get("status", "no_response"))
            if status == "invalid_json":
                self._llm_invalid_response_count += 1
            else:
                self._llm_no_response_count += 1
            event = {
                "generation": self._t,
                "status": status,
                "detail": str(status_info.get("detail", "empty policy response")),
                "operator_before": before_preference,
                "operator_after": before_preference,
                "changed": [],
                **status_info,
            }
            self._fail_llm_policy(status, event["detail"], event)
            return

        try:
            proposal_status, proposal, changed, detail = self._validate_llm_proposal(raw)
        except ValueError as exc:
            self._llm_schema_rejected_count += 1
            event = {
                "generation": self._t,
                "status": "schema_rejected",
                "detail": str(exc),
                "operator_before": before_preference,
                "operator_after": before_preference,
                "changed": [],
                **status_info,
            }
            self._fail_llm_policy("schema_rejected", str(exc), event)
            return

        self._apply_llm_proposal(proposal)
        self._llm_last_status = proposal_status
        if proposal_status == "applied":
            self._llm_applied_count += 1
        else:
            self._llm_noop_count += 1
        latency = float(status_info.get("latency_ms", 0.0) or 0.0)
        self._llm_latency_ms.append(latency)
        self._record_llm_event({
            "generation": self._t,
            "status": proposal_status,
            "detail": detail,
            "operator_before": before_preference,
            "operator_after": self._llm_preference,
            "changed": changed,
            "params_before": before_params,
            "params_after": dict(self._operator_params),
            "apd_alpha_before": before_alpha,
            "apd_alpha_after": self.apd_alpha,
            **status_info,
        })

    def _apply_llm_proposal(self, proposal: Mapping[str, Any]) -> None:
        """Apply a prevalidated LLM proposal to policy state."""
        if "operator_choice" in proposal:
            self._llm_preference = str(proposal["operator_choice"])
        for name, value in dict(proposal.get("operator_params", {})).items():
            self._operator_params[name] = float(value)
        if "apd_alpha" in proposal:
            self.apd_alpha = float(proposal["apd_alpha"])

        # Adaptive reference frequency (accelerate on difficult fronts).
        if self._stagnation > 2:
            self.adapt_period = max(10, self.adapt_period // 2)
        else:
            self.adapt_period = 50

    def llm_telemetry(self) -> Dict[str, Any]:
        """Return serializable telemetry for a completed MaACO run."""
        latency = (sum(self._llm_latency_ms) / len(self._llm_latency_ms)
                   if self._llm_latency_ms else 0.0)
        return {
            "policy_mode": "required_local_llm" if self.use_llm else "disabled",
            "llm_enabled": int(self.use_llm),
            "llm_calls": int(self._llm_call_count),
            "llm_applied": int(self._llm_applied_count),
            "llm_no_response": int(self._llm_no_response_count),
            "llm_invalid_response": int(self._llm_invalid_response_count),
            "llm_schema_rejected": int(self._llm_schema_rejected_count),
            "llm_noop": int(self._llm_noop_count),
            "llm_failures": int(self._llm_failed_count),
            "llm_mean_latency_ms": float(latency),
            "llm_last_status": str(self._llm_last_status),
            "llm_events": list(self._llm_events),
        }

    # ------------------------------------------------------------------
    # APD Environmental Selection (RVEA-style)
    # ------------------------------------------------------------------

    def _apd_selection(self, F: np.ndarray) -> np.ndarray:
        n, m = F.shape
        if self.ref_dirs.shape[1] != m:
            self.ref_dirs = np.asarray(
                get_reference_directions(
                    "das-dennis", m,
                    n_partitions=getattr(self, '_ref_partitions', 12)),
                dtype=float,
            )
        target = min(self.pop_size, self.ref_dirs.shape[0])
        if n <= target:
            return np.arange(n)

        # Translation-invariant normalization
        f_min = F.min(axis=0)
        Fn = F - f_min
        denom = np.maximum(Fn.max(axis=0), 1e-12)
        Fn = Fn / denom

        ref_dirs = self.ref_dirs.copy()
        ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        ref_dirs = ref_dirs / np.maximum(ref_norms, 1e-12)

        fn_norms = np.linalg.norm(Fn, axis=1, keepdims=True)
        safe_norms = np.where(fn_norms < 1e-12, 1.0, fn_norms)
        v = Fn / safe_norms
        v = np.where(fn_norms < 1e-12, 0.0, v)

        cos = np.clip(v @ ref_dirs.T, -1.0, 1.0)
        assigned = np.argmax(cos, axis=1)
        theta = np.arccos(cos[np.arange(n), assigned])
        norm = np.linalg.norm(Fn, axis=1)

        t_max = max(1, int(getattr(self, "n_gen", 1000) or 1000))
        t_hat = min(1.0, max(0.05, self._t / t_max))
        penalty = 1.0 + float(m) * (t_hat ** self.apd_alpha) * theta
        apd = penalty * norm

        # One champion per active reference vector (guarantees coverage
        # across all niches, matching NSGA-III's niching property)
        niche_champions: List[int] = []
        K = ref_dirs.shape[0]
        for ref_k in range(K):
            mask = assigned == ref_k
            if np.any(mask):
                candidates = np.where(mask)[0]
                best = candidates[np.argmin(apd[candidates])]
                niche_champions.append(int(best))

        if len(niche_champions) <= target:
            keep = list(niche_champions)
            # Fill remaining slots with lowest APD non-champions
            if len(keep) < target:
                remaining = [i for i in np.argsort(apd) if int(i) not in set(keep)]
                keep.extend(int(i) for i in remaining[: target - len(keep)])
        else:
            # More active niches than target: keep champions with smallest APD
            order = np.argsort([apd[c] for c in niche_champions])
            keep = [niche_champions[i] for i in order[:target]]

        return np.asarray(keep[:target], dtype=int)

    # ------------------------------------------------------------------
    # Reference Vector Adaptation
    # ------------------------------------------------------------------

    def _adapt_reference_directions(self,
                                    F: np.ndarray,
                                    ref_dirs: np.ndarray) -> np.ndarray:
        if F.shape[0] < 2:
            return ref_dirs

        Fn = F - F.min(axis=0)
        denom = np.maximum(Fn.max(axis=0), 1e-12)
        Fn = Fn / denom
        v = Fn / np.maximum(np.linalg.norm(Fn, axis=1, keepdims=True), 1e-12)
        ref_v = ref_dirs / np.maximum(
            np.linalg.norm(ref_dirs, axis=1, keepdims=True), 1e-12)

        cos = v @ ref_v.T
        assigned = np.argmax(cos, axis=1)

        active_set = set(assigned.tolist())
        inactive = [i for i in range(ref_dirs.shape[0]) if i not in active_set]
        if not inactive:
            return ref_dirs

        new_dirs = ref_dirs.copy()
        n_inactive = len(inactive)
        n_avail = min(n_inactive, F.shape[0])
        if n_avail > 0:
            sample_idx = self._rng.choice(F.shape[0], size=n_avail,
                                         replace=False)
            sample = Fn[sample_idx]
            sample = np.maximum(sample, 1e-6)
            sample = sample / sample.sum(axis=1, keepdims=True)
            for i, idx in enumerate(inactive[:n_avail]):
                new_dirs[idx] = sample[i]
        return new_dirs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nsga2_ranks(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NSGA-II non-dominated ranking with crowding distance tiebreak."""
    n = F.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    ranks = np.empty(n, dtype=int)
    crowding = np.zeros(n, dtype=float)
    for rank_idx, front in enumerate(fronts):
        ranks[front] = rank_idx
        if len(front) > 1:
            crowding[front] = calc_crowding_distance(F[front])
        else:
            crowding[front[0]] = np.inf
    return ranks, crowding


def _fast_hv_estimate(F: np.ndarray) -> float:
    """Fast hypervolume proxy used exclusively for the bandit reward signal.

    Avoids calling the full exact HV calculator on every generation.
    Computes hypervolume over normalized objectives with a fixed reference
    at [1.1, ..., 1.1].
    """
    if F.size == 0 or F.shape[0] == 0:
        return 0.0
    n, m = F.shape
    f_min = F.min(axis=0)
    denom = np.maximum(F.max(axis=0) - f_min, 1e-12)
    Fn = (F - f_min) / denom
    # Filter to non-dominated points
    try:
        nd_idx = NonDominatedSorting().do(Fn, only_non_dominated_front=True)
        Fn_nd = Fn[nd_idx]
    except Exception:
        Fn_nd = Fn
    if Fn_nd.size == 0:
        return 0.0
    # Proxy: sum of distance from reference [1.1, ..., 1.1]
    # (higher = better coverage toward ideal [0, ..., 0])
    ref = np.ones(m) * 1.1
    vol = np.sum(np.prod(np.maximum(0.0, ref - Fn_nd), axis=1))
    return float(vol)
