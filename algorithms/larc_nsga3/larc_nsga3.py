"""LARC-NSGA3: Two-Tier Hybrid Evolutionary Algorithm with Local Decision-Model Strategy Controller.

Authors
-------
Thiago Santos, Sebastiao Xavier (UFOP / METISBr, 2026).

Paper
-----
"Strict Local-LLM Discrete Action Control for Adaptive NSGA-III in Many-Objective Optimization"
(manuscript under revision; venue-neutral reference).

Architecture
------------
1. Tier 1 (Deterministic Heuristic): State-aware reactive decision tree executed every generation.
2. Tier 2 (Bounded Supervisor): a pluggable controller queried at bounded intervals
   (B <= llm_max_calls_abs) on the schedule of ``_should_query_llm``. Controllers:

   * ``heuristic``      -- no Tier-2 queries at all (Tier 1 only, no model is loaded);
   * ``random``         -- uniform draw over the trap-filtered allowed set;
   * ``bandit``         -- UCB1 over the EMA action credit
     (score = ema[a] + c*sqrt(ln(1+t)/(1+n_a)), c = ``_BANDIT_UCB_C`` = 0.5);
   * ``laya_blind``     -- local open-weights decision model (Laya, convaiinnovations/laya,
     Apache-2.0) through the closed-set ``choice`` primitive with the legacy prompt
     (``controller="laya"`` is an alias);
   * ``laya_informed``  -- same model, English prompt carrying the numeric state, the EMA
     credit, the last three action->outcome records and operator-level action descriptions.

   All Tier-2 controllers share the same schedule, call budget, trap filter, anti-collapse
   guardrail and ``decision_hold``; only the decision rule differs.
3. State-Diagnostic Traps: Filters out catastrophic exploration on degenerate/multimodal fronts.
4. Graceful Fallback: In case of model unavailability, malformed output, or confidence below
   ``min_llm_confidence``, the controller falls back to the Tier-1 heuristic action
   (source label ``error_fallback``), eliminating survival bias.
5. Zero-Pressure Random Mating: Prevents boundary destruction in high-dimensional spaces (M >= 5).

Per-generation trace
--------------------
``action_history`` holds one record per generation with keys ``generation``, ``action``,
``source``, ``controller``, ``confidence`` and ``reason_code``. ``source`` is one of
``heuristic`` (Tier 1 between Tier-2 queries), ``tier2`` (accepted Tier-2 decision),
``tier2_hold`` (Tier-2 decision kept in force by ``decision_hold``), ``trap_fallback``
(Tier-2 choice, or held choice, rejected by the trap filter), ``error_fallback`` (client
error, malformed answer or confidence below the floor -> Tier-1 heuristic) and ``guardrail``
(anti-collapse override). ``tier2_io_log`` stores, per Tier-2 decision, the exact text,
criteria, instructions and language routed to Laya plus the raw answer.

Revision 2026-09 changes
------------------------
Behavioural changes relative to the pre-revision code for ``LARC_NSGA3()`` with default
arguments (controller ``"laya"`` == ``"laya_blind"``, ``decision_hold=1``):

* Bug fix (subpop): ``subpop`` was a no-op because environmental selection always returns
  exactly ``pop_size`` individuals. It now swaps, for every reference niche that is empty in
  the survivors but has candidates in the parent+offspring pool, the best-angle candidate
  in, removing a member of the most crowded niche (only niches with >= 2 members donate).
  Population size is unchanged.
* Bug fix (ref): ``ref_dirs`` is deep-copied at construction and at setup, and the
  adaptation never mutates arrays owned by the caller (previously the caller's array,
  shared by the benchmark harness with NSGA3/RVEA/MOEAD, was modified in place).
  ``ref_dirs_trajectory`` records (generation, sum |ref_dirs - ref_dirs_initial|).
* Source labels in ``action_history`` changed: ``llm`` -> ``tier2``; the trap-filtered
  Tier-2 fallback is ``trap_fallback`` and error/low-confidence fallbacks are
  ``error_fallback`` (both were ``heuristic`` before). Records also carry ``controller``.
* The Laya client is constructed lazily on the first Tier-2 query instead of in
  ``__init__``; a construction failure is therefore reported as ``error_fallback``
  generations (counted in ``graceful_fallbacks``) instead of an exception at construction.
* ``llm_min_calls_abs`` may now be 0 (previously clamped to >= 1) so that a Tier-2 budget
  B = 0 (pure heuristic) can be requested; the default (6) is unchanged.
* The ``controller`` argument is honoured (it was hard-coded to ``"laya"``).
* Exact Tier-2 budget (``tier2_schedule``): when ``llm_min_calls_abs == llm_max_calls_abs
  == B`` (default ``tier2_schedule="auto"``) the Tier-2 controller is queried on a *fixed*
  schedule that realises exactly B queries: query k (k = 1..B) becomes due once the consumed
  evaluation fraction reaches (k - 0.5)/B, and at most one query runs per generation (a due
  query deferred by ``decision_hold`` or by an earlier query in the same generation is run
  at the next generation). The schedule ignores ``llm_min_gap``, ``llm_interval`` and the
  state triggers, so it is independent of the search trajectory and identical for every
  controller on the same instance. B is realised whenever the run has at least B
  generations after the scheduled points (``llm_call_stats["total_queries"]`` records the
  realised count, ``tier2_budget`` the requested one). With the defaults (6 / 15) the
  schedule stays ``adaptive`` (pre-revision behaviour, B is only a cap).
* ``state_feature_mask`` also neutralises masked outcome fields in the action->outcome
  history shown to Tier-2 (``improvement``, ``entropy_delta``, ``nd_delta``).
* ``state_feature_mask`` (repair r2): a masked feature now reaches no decision input. The
  trap filter, the anti-collapse guardrail, the EMA credit assignment (and its warm start)
  run on the masked view, so the allowed-action set handed to Tier-1 and Tier-2 no longer
  encodes masked phase / budget / stagnation / empty-niche / ND-ratio values; masking
  ``budget_ratio`` or ``phase`` also hides ``generation`` (and the generation numbers of
  the history records), which would otherwise reveal budget progress. Only the query
  *schedule* still reads the true state (the fixed schedule depends on the consumed
  evaluation fraction alone). Without a mask nothing changes.
* Bug fix (single evaluation, 2026-09-30): the variation operator receives the decision
  matrix, so the offspring are evaluated once by the framework evaluator; with ``var`` the
  offspring used to be evaluated inside the operator and again after the extra mutation,
  which made the evaluations consumed per generation depend on the action.
* Bug fix (mating, 2026-09-30): zero-pressure mating is now uniform for every action
  (``_mating_pool``). The former binary tournament on an all-zero fitness sent ties to the
  lower index (about 19% of the parents from the first tenth of the population and 1% from
  the last tenth at N = 100); ``var`` already drew uniformly. With constraints each parent
  wins a binary tournament on the total violation, ties going to the first uniform draw.
  Test: ``experiments/tests/test_harness_f2.py::test_larc_mating_pool_is_uniform...``.
* Bug fix (credit, 2026-09-30): the EMA credit starts at t = 1. The action chosen at
  initialisation was credited with deltas between s_0 (measured on P_0) and s_1 (measured on
  U_0 = P_0 + Q_0), two different bases. Test: ``...::test_larc_credit_starts_at_second_generation``.
The legacy (``laya_blind``) prompt payload, system prompt and Laya text/criteria/lang are
byte-for-byte identical to the pre-revision code.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as _np
np = _np  # host NumPy: this algorithm's state handling is CPU-side
from core.tensor.backend import to_numpy
from core.population import Population

from algorithms.nsga3.nsga3 import (
    NSGA3,
    _constraint_violation,
    _environmental_selection,
    _population_constraints,
    _population_objectives,
    _update_zmin,
)
from algorithms.community_utils.moead_family import rng_from_algo
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from core.llm.laya_policy import LayaPolicyClient, build_laya_request
logger = logging.getLogger(__name__)

ALGORITHM_FLAGS = {
    "LARC_NSGA3": {"multi", "many", "real", "integer"},
    "LARC_NSGA3_NoTraps": {"multi", "many", "real", "integer"},
    "Heuristic_NSGA3": {"multi", "many", "real", "integer"},
}

_ACTIONS = ("conv", "div", "var", "ref", "subpop", "pref")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LLM_LOG_PATH = _PROJECT_ROOT / "logs" / "larc_nsga3_usage.jsonl"

#: Admissible controllers. ``laya`` is kept as a legacy alias of ``laya_blind``.
_CONTROLLERS = ("heuristic", "random", "bandit", "laya_blind", "laya_informed")
_CONTROLLER_ALIASES = {"laya": "laya_blind"}
_LAYA_CONTROLLERS = ("laya_blind", "laya_informed")

#: Exploration constant of the UCB1 bandit controller.
_BANDIT_UCB_C = 0.5
#: Salt mixed with the algorithm seed to derive the independent Tier-2 RNG stream.
_TIER2_SEED_SALT = 0x7A1E2
#: Number of action->outcome records shown to the informed controller.
_INFORMED_HISTORY = 3
#: Tier-2 query schedules: ``fixed`` realises exactly B queries (see module docstring),
#: ``adaptive`` is the pre-revision state-triggered schedule with B as a cap, ``auto``
#: selects ``fixed`` iff ``llm_min_calls_abs == llm_max_calls_abs``.
_TIER2_SCHEDULES = ("auto", "fixed", "adaptive")

#: Neutral constants that replace masked state features (``state_feature_mask``).
#: They are chosen so that no Tier-1 rule, no trap and no narrative threshold is triggered
#: by a masked feature: entropy 0.5 (> 0.45 low-diversity and > 0.20 collapse thresholds),
#: ND ratio 0.5 (between the 0.20 multimodal and 0.70 "very high" thresholds),
#: empty niches 0.2 (< 0.40 degenerate-front threshold), zero improvement/deltas,
#: no stagnation, mid budget, "middle" phase and generation 0. Masking applies to every
#: decision input (repair r2): the Tier-1 heuristic, the Tier-2 controller, the trap filter
#: (so the allowed-action set cannot encode a masked feature), the anti-collapse guardrail
#: and the EMA credit assignment / warm start (the credit shown to Tier-2 and used by
#: Tier-1 and the bandit is computed from the masked view). Masked ``improvement``,
#: ``entropy_delta`` and ``non_dominated_delta`` are also neutralised in the
#: action->outcome history records shown to Tier-2 (``_HISTORY_FIELD_OF_FEATURE``).
#: Masking ``budget_ratio`` or ``phase`` also masks ``generation`` (``_IMPLIED_MASK``),
#: including the generation numbers of the history records. Only the Tier-2 query
#: *schedule* keeps reading the true state (it decides when, never what).
_NEUTRAL_STATE_VALUES: Dict[str, Any] = {
    "budget_ratio": 0.5,
    "phase": "middle",
    "crowding_entropy": 0.5,
    "non_dominated_ratio": 0.5,
    "angle_dispersion": 0.5,
    "angle_dispersion_delta": 0.0,
    "empty_niches": 0.2,
    "best_norm_sum": 0.0,
    "improvement": 0.0,
    "entropy_delta": 0.0,
    "non_dominated_delta": 0.0,
    "stagnation": 0,
    "generation": 0,
}

#: Features hidden together with a masked feature: the generation index reveals budget
#: progress, so it is hidden whenever ``budget_ratio`` or ``phase`` is masked.
_IMPLIED_MASK: Dict[str, tuple] = {"budget_ratio": ("generation",), "phase": ("generation",)}

#: Map from maskable state features to the matching field of an action->outcome history
#: record (``_state_action_outcomes``). When one of these features is masked, the field is
#: replaced by the feature's neutral constant in every history record sent to Tier-2, so
#: the true (masked) value cannot reach the controller through the history channel.
_HISTORY_FIELD_OF_FEATURE = {
    "improvement": "improvement",
    "entropy_delta": "entropy_delta",
    "non_dominated_delta": "nd_delta",
}

# Legacy (laya_blind) prompt material -- must stay byte-for-byte identical.
_LEGACY_ACTION_SEMANTICS = {
    "conv": "Balanced exploration-exploitation with standard polynomial mutation and SBX crossover.",
    "div": "Diversity expansion via high-entropy mating and widened distribution index.",
    "var": "Decision-space perturbation using variable-classification Gaussian mutation to escape stagnation.",
    "ref": "Reference vector adaptation shifted toward current population objective centroids.",
    "subpop": "Injection of under-represented niche individuals from offspring pool into population.",
    "pref": "Late-stage fine-grained boundary polishing with aggressive exploitation parameter scaling.",
}
_LEGACY_SYSTEM_PROMPT = (
    "You are the discrete controller of LARC-NSGA3 for many-objective optimization. "
    "Your goal is to arbitrate search phases based on population state metrics and recent action rewards. "
    "Available actions and semantics: "
    "- conv: standard balanced exploration-exploitation (SBX crossover, polynomial mutation). "
    "- div: diversity expansion across poorly covered niches via high-entropy mating. "
    "- var: decision-space perturbation using variable-classification mutation to escape stagnation. "
    "- ref: reference vector adaptation shifting directions toward current population centroids. "
    "- subpop: injection of candidates from under-represented niches. "
    "- pref: fine-grained late-stage boundary polishing and tight local convergence. "
    "You must select exactly one action from allowed_actions that best fits the current search state. "
    "Output ONLY a valid JSON object matching: "
    '{"action": "<chosen_action>", "confidence": <0.0-1.0>, "reason_code": "<short_reason>"}'
)
_OUTPUT_SCHEMA = {"action": "string", "confidence": "float", "reason_code": "string"}

# Informed-controller "use when" cues (appended to the operator description of each action).
#: Reward saturation scales (tanh(scale * delta)); spec 001 FR-007. The defaults are the values used
#: since the first release; they are constructor parameters so their sensitivity can be measured
#: (Reviewer 13.9/13.29). Scale ~ 1/typical |delta| puts one typical step near tanh's linear-to-
#: saturation knee: one-generation improvement ~0.03 -> 30; ND-ratio/entropy deltas ~0.15 -> 6;
#: angular-dispersion delta ~0.12 -> 8, negated because lower dispersion is better.
DEFAULT_REWARD_SCALES = {"improvement": 30.0, "nd": 6.0, "entropy": 6.0, "angle": -8.0}
#: Per-action weights (improvement, nd, entropy, angle); each row sums to 1.
DEFAULT_REWARD_WEIGHTS = {
    "conv": (0.55, 0.25, 0.10, 0.10),
    "div": (0.15, 0.10, 0.60, 0.15),
    "ref": (0.35, 0.15, 0.20, 0.30),
    "subpop": (0.25, 0.15, 0.35, 0.25),
    "pref": (0.50, 0.30, 0.10, 0.10),
    "var": (0.30, 0.15, 0.40, 0.15),
}

#: Default parameters for reference vector adaptation (action 'ref')
DEFAULT_REF_GAMMA = 0.15      #: Elastic canonical relaxation rate for inactive niches
DEFAULT_REF_ALPHA_MAX = 0.08  #: Maximum adaptation step towards local niche centroid
DEFAULT_REF_BETA = 0.02       #: Static anchor spring to initial canonical reference direction

_USE_WHEN = {
    "conv": "the search is improving steadily.",
    "div": "niche entropy is low or many niches are empty early.",
    "var": "the search stagnates with few non-dominated solutions.",
    "ref": "many niches stay empty mid-run (irregular front).",
    "subpop": "empty niches have candidates in the offspring pool.",
    "pref": "late phase and most solutions are non-dominated.",
}
_EFFECT_SHORT = {
    "conv": "standard balanced variation.",
    "div": "wider offspring spread.",
    "var": "random kicks on 15% of variables.",
    "ref": "adapt reference directions on the unit simplex.",
    "subpop": "swap candidates into empty niches.",
    "pref": "fine local polishing.",
}
_INFORMED_OPTION_MAX_TOKENS = 26


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


def _host_array(values: Any, dtype: Any = float) -> _np.ndarray:
    """Return a private host-NumPy copy (never a view of the caller's buffer)."""
    return _np.array(to_numpy(values), dtype=dtype, copy=True)


def _jsonable(obj: Any) -> Any:
    def _default(o: Any) -> Any:
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    try:
        return json.loads(json.dumps(obj, default=_default))
    except Exception:
        return str(obj)


class LARC_NSGA3(NSGA3):
    """NSGA-III with Two-Tier Bounded Controller (see module docstring)."""

    ALGO_FLAGS = {"multi", "many", "real", "integer"}
    OBJECTIVE_SCOPE = "many"
    LOG_ALGORITHM_NAME = "LARC_NSGA3"
    DEFAULT_LOG_PATH = _DEFAULT_LLM_LOG_PATH

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: Any = None,
        sampling: Any = None,
        controller: str = "laya",
        llm_client: Optional[Any] = None,
        llm_log_path: Any = None,
        policy_temperature: float = 0.0,
        llm_interval: int = 15,
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
        decision_hold: int = 1,
        state_feature_mask: Optional[Sequence[str]] = None,
        tier2_seed: Optional[int] = None,
        tier2_schedule: str = "auto",
        reward_scales: Optional[Dict[str, float]] = None,
        reward_weights: Optional[Dict[str, Sequence[float]]] = None,
        ref_gamma: float = DEFAULT_REF_GAMMA,
        ref_alpha_max: float = DEFAULT_REF_ALPHA_MAX,
        ref_beta: float = DEFAULT_REF_BETA,
        **kwargs: Any,
    ) -> None:
        super().__init__(pop_size=pop_size, ref_dirs=ref_dirs, sampling=sampling, **kwargs)
        # A2: never alias the caller's reference-direction array.
        if self.ref_dirs is not None:
            self.ref_dirs = _host_array(self.ref_dirs)
        self._ref_dirs_initial: Optional[_np.ndarray] = None if self.ref_dirs is None else self.ref_dirs.copy()
        self._extreme_ref_mask: Optional[_np.ndarray] = (
            None if self._ref_dirs_initial is None else (
                _np.isclose(_np.max(self._ref_dirs_initial, axis=1), 1.0) &
                _np.isclose(_np.sum(self._ref_dirs_initial, axis=1), 1.0)
            )
        )
        self.ref_dirs_trajectory: list[tuple[int, float]] = []

        self.enable_traps = bool(enable_traps)
        # FR-007: reward constants are parameters (defaults = first-release values).
        unknown = set(reward_scales or {}) - set(DEFAULT_REWARD_SCALES)
        if unknown:
            raise ValueError(f"unknown reward_scales keys: {sorted(unknown)}")
        self.reward_scales = {**DEFAULT_REWARD_SCALES, **{k: float(v) for k, v in (reward_scales or {}).items()}}
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
        for act, w in (reward_weights or {}).items():
            if act not in DEFAULT_REWARD_WEIGHTS or len(tuple(w)) != 4:
                raise ValueError(f"reward_weights[{act!r}] must be a 4-tuple for a known action")
            self.reward_weights[act] = tuple(float(x) for x in w)

        if not (0.0 <= float(ref_gamma) <= 1.0):
            raise ValueError(f"ref_gamma must be in [0, 1], got {ref_gamma!r}")
        if not (0.0 <= float(ref_alpha_max) <= 1.0):
            raise ValueError(f"ref_alpha_max must be in [0, 1], got {ref_alpha_max!r}")
        if not (0.0 <= float(ref_beta) <= 1.0):
            raise ValueError(f"ref_beta must be in [0, 1], got {ref_beta!r}")
        if float(ref_alpha_max) + float(ref_beta) > 1.0:
            raise ValueError("ref_alpha_max + ref_beta must not exceed 1.0")
        self.ref_gamma = float(ref_gamma)
        self.ref_alpha_max = float(ref_alpha_max)
        self.ref_beta = float(ref_beta)

        ctrl = str(controller if controller is not None else "laya").strip().lower()
        ctrl = _CONTROLLER_ALIASES.get(ctrl, ctrl)
        if ctrl not in _CONTROLLERS:
            raise ValueError(f"Unknown LARC controller {controller!r}; expected one of {_CONTROLLERS} (or 'laya').")
        if not bool(enable_llm):
            ctrl = "heuristic"
        self.controller = ctrl
        # ``enable_llm`` now means "Tier-2 controller active" (kept for backward compatibility).
        self.enable_llm = ctrl != "heuristic"
        # Only Laya controllers use a model client; it is built lazily on the first query.
        self.llm_client = llm_client if ctrl in _LAYA_CONTROLLERS else None

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
        self.llm_min_calls_abs = int(max(0, llm_min_calls_abs))
        self.llm_max_calls_abs = int(max(self.llm_min_calls_abs, llm_max_calls_abs))
        schedule = str(tier2_schedule if tier2_schedule is not None else "auto").strip().lower()
        if schedule not in _TIER2_SCHEDULES:
            raise ValueError(f"Unknown tier2_schedule {tier2_schedule!r}; expected one of {_TIER2_SCHEDULES}.")
        if schedule == "auto":
            schedule = "fixed" if self.llm_min_calls_abs == self.llm_max_calls_abs else "adaptive"
        self.tier2_schedule = schedule

        self.decision_hold = int(max(1, decision_hold))
        mask = tuple(str(k) for k in (state_feature_mask or ()))
        unknown = [k for k in mask if k not in _NEUTRAL_STATE_VALUES]
        if unknown:
            raise ValueError(f"Unknown state features in state_feature_mask: {unknown}; "
                             f"maskable features: {sorted(_NEUTRAL_STATE_VALUES)}")
        self.state_feature_mask: tuple[str, ...] = mask
        hidden = list(mask)
        for key in mask:
            hidden.extend(k for k in _IMPLIED_MASK.get(key, ()) if k not in hidden)
        #: masked features plus the ones they imply (``_IMPLIED_MASK``)
        self.hidden_state_features: tuple[str, ...] = tuple(hidden)
        self.tier2_seed = None if tier2_seed is None else int(tier2_seed)
        self._tier2_rng: Optional[_np.random.Generator] = None

        self.current_action = "conv"
        self.action_history: list[dict[str, Any]] = []
        self.state_history: list[dict[str, Any]] = []
        self.tier2_io_log: list[dict[str, Any]] = []
        self._pending_io: Optional[dict[str, Any]] = None
        self._best_norm_sum_history: list[float] = []
        self._action_reward_ema = {action: 0.0 for action in _ACTIONS}
        self._action_counts = {action: 0 for action in _ACTIONS}
        self._bandit_t = 0
        self._bandit_counts = {action: 0 for action in _ACTIONS}
        self._hold_action: Optional[str] = None
        self._hold_remaining = 0
        self._hold_confidence = 0.0
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
            "hold_generations": 0,
            "hold_trap_fallbacks": 0,
            "below_confidence_floor": 0,
            "client_constructions": 0,
            "subpop_swaps": 0,
            "tier2_schedule": self.tier2_schedule,
            "tier2_budget": int(self._llm_call_budget()),
        }

    # ------------------------------------------------------------------ setup / RNG
    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.ref_dirs = _host_array(self.ref_dirs)
        self._ref_dirs_initial = self.ref_dirs.copy()
        if self._ref_dirs_initial is not None:
            self._extreme_ref_mask = (
                _np.isclose(_np.max(self._ref_dirs_initial, axis=1), 1.0) &
                _np.isclose(_np.sum(self._ref_dirs_initial, axis=1), 1.0)
            )
        else:
            self._extreme_ref_mask = None
        self.ref_dirs_trajectory = []
        self._tier2_rng = None  # re-derive from the (possibly new) algorithm seed
        self.llm_call_stats["tier2_budget"] = int(self._llm_call_budget())

    def _tier2_generator(self) -> _np.random.Generator:
        """Independent RNG used only by the random/bandit controllers."""
        if self._tier2_rng is None:
            if self.tier2_seed is not None:
                seq = _np.random.SeedSequence(int(self.tier2_seed))
            else:
                base = getattr(self, "seed", None)
                if base is None:
                    seq = _np.random.SeedSequence()
                else:
                    seq = _np.random.SeedSequence([int(base) & 0xFFFFFFFF, _TIER2_SEED_SALT])
            self._tier2_rng = _np.random.Generator(_np.random.PCG64(seq))
        return self._tier2_rng

    def _get_llm_client(self) -> Any:
        if self.llm_client is None:
            self.llm_call_stats["client_constructions"] += 1
            # Exclusive Tier-2 local decision model: Laya 322M (multilingual)
            self.llm_client = LayaPolicyClient(
                checkpoint="convaiinnovations/laya",
                subfolder="multilingual",
                prefer_local_import=True,
            )
        return self.llm_client

    # ------------------------------------------------------------------ main loop
    def _initialize_advance(self, infills=None, **kwargs: Any) -> None:
        super()._initialize_advance(infills=infills, **kwargs)
        if self.pop is not None and len(self.pop) > 0:
            state = self._population_state(self.pop)
            self._warm_start_ema(self._decision_view(state))
            allowed = tuple(_ACTIONS)
            initial_action = self._heuristic_action(self._decision_view(state), allowed)
            self.current_action = initial_action
            self._action_streak = 1
            record = {
                "generation": 0,
                "action": initial_action,
                "confidence": 1.0,
                "reason_code": "initial_heuristic",
                "source": "heuristic",
                "controller": self.controller,
            }
            self.action_history.append(record)
            self.state_history.append(state)

    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return super()._infill()

        rng = rng_from_algo(self)
        mating = self._mating_pool(rng)

        params = self._variation_parameters(self.current_action)
        # decision matrix in, decision matrix out: the offspring are evaluated once, by the framework evaluator
        # (a population argument would make the operator evaluate them itself, and the "var" action would then pay
        # a second evaluation after its extra mutation, making the budget consumed per generation action-dependent)
        X_off = OperatorGA(self.problem, np.asarray(self.pop[mating].get("X"), dtype=float), Parameter=params, rng=rng)
        offspring = Population.new("X", np.asarray(X_off, dtype=float))
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
            self._record_ref_dirs_trajectory()
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
        # Credit guard (t >= 1): the first generation is not credited, because its previous state s_0 was measured on
        # the initial population P_0 whereas every later state is measured on the merged pool U_t = P_t + Q_t.
        if self.action_history and len(self.state_history) >= 2:
            previous = self.action_history[-1]
            self._assign_credit(previous, self.state_history[-1], state)
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
            "source": str(decision.get("source", "tier2")),
            "controller": self.controller,
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

    def _decision_view(self, state: dict[str, Any]) -> dict[str, Any]:
        """State as seen by the decision rules (masked and implied features -> neutral constants)."""
        if not self.hidden_state_features:
            return state
        view = dict(state)
        for key in self.hidden_state_features:
            view[key] = _NEUTRAL_STATE_VALUES[key]
        return view

    def _assign_credit(self, previous: dict[str, Any], prev_state: dict[str, Any], cur_state: dict[str, Any]) -> None:
        """EMA credit of ``previous`` computed from the masked views (identity without a mask)."""
        self._credit_action(previous, self._decision_view(prev_state), self._decision_view(cur_state))

    def _tier2_history(self, n: int) -> list[dict[str, Any]]:
        """Last ``n`` action->outcome records as shown to Tier-2 (masked fields neutralised).

        Without a mask the records are returned as stored (legacy byte-identical prompt).
        With a mask, copies are returned whose masked outcome fields hold the neutral
        constant, so a masked feature never reaches Tier-2 through the history channel.
        """
        history = self._state_action_outcomes[-n:] if self._state_action_outcomes else []
        fields = {
            _HISTORY_FIELD_OF_FEATURE[k]: _NEUTRAL_STATE_VALUES[k]
            for k in self.hidden_state_features
            if k in _HISTORY_FIELD_OF_FEATURE
        }
        if "generation" in self.hidden_state_features:
            fields["gen"] = _NEUTRAL_STATE_VALUES["generation"]
        if not fields:
            return history
        out: list[dict[str, Any]] = []
        for rec in history:
            rec = dict(rec)
            for field, neutral in fields.items():
                if field in rec:
                    rec[field] = neutral
            out.append(rec)
        return out

    # ------------------------------------------------------------------ decision logic
    def _select_policy_action(self, state: dict[str, Any]) -> dict[str, Any]:
        # Every decision input (traps, guardrail, rules) sees the masked view; only the
        # query schedule (``_should_query_llm``) reads the true state.
        view = self._decision_view(state)
        allowed = tuple(a for a in self._allowed_actions(view) if a in _ACTIONS)
        if not allowed:
            allowed = _ACTIONS

        # Decision hold: a Tier-2 decision stays in force, re-checked against the traps.
        if self._hold_remaining > 0 and self._hold_action is not None:
            self._hold_remaining -= 1
            if self._hold_action in allowed:
                decision = {
                    "action": self._hold_action,
                    "confidence": float(self._hold_confidence),
                    "reason_code": "tier2_decision_hold",
                    "source": "tier2_hold",
                }
                decision = self._guardrail_decision(decision, view, allowed)
                if decision.get("source") == "tier2_hold":
                    self.llm_call_stats["hold_generations"] += 1
                return decision
            self.llm_call_stats["hold_trap_fallbacks"] += 1
            return {
                "action": self._heuristic_action(view, allowed),
                "confidence": 0.75,
                "reason_code": "hold_action_trap_filtered",
                "source": "trap_fallback",
            }

        if self.enable_llm and self._should_query_llm(state, allowed):
            decision = self._query_tier2(state, view, allowed)
            controller_action = str(decision.get("action", ""))
            decision = self._guardrail_decision(decision, view, allowed)
            if self._pending_io is not None:
                self._pending_io["controller_action"] = controller_action
                self._pending_io["final_action"] = str(decision.get("action", ""))
                self._pending_io["source"] = str(decision.get("source", ""))
                self._pending_io["reason_code"] = str(decision.get("reason_code", ""))
                self._pending_io = None
            if decision.get("source") == "tier2" and self.decision_hold > 1:
                self._hold_action = str(decision["action"])
                self._hold_remaining = self.decision_hold - 1
                self._hold_confidence = float(decision.get("confidence", 0.0))
            return decision

        self.llm_call_stats["tier1_heuristic_count"] += 1
        heuristic_action = self._heuristic_action(view, allowed)
        return {
            "action": heuristic_action,
            "confidence": 0.75,
            "reason_code": "heuristic_between_llm",
            "source": "heuristic",
        }

    def _query_tier2(self, state: dict[str, Any], view: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
        """Run one Tier-2 decision and accumulate its wall-clock cost in ``tier2_time_s``.

        The time includes lazy client construction (model load) on the first Laya query;
        ``tier2_query_times_s`` keeps the per-query values so load and inference can be
        separated in the analysis (Reviewer 13.19: report the real cost of the LLM).
        """
        t0 = time.perf_counter()
        try:
            return self._query_tier2_impl(state, view, allowed)
        finally:
            dt = time.perf_counter() - t0
            self.tier2_time_s = float(getattr(self, "tier2_time_s", 0.0)) + dt
            self.tier2_query_times_s = list(getattr(self, "tier2_query_times_s", [])) + [dt]

    def _query_tier2_impl(self, state: dict[str, Any], view: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
        self._llm_queries += 1
        self.llm_call_stats["total_queries"] += 1
        self._last_llm_generation = int(state.get("generation", self._generation_index()))
        if self.controller == "random":
            return self._random_decision(state, allowed)
        if self.controller == "bandit":
            return self._bandit_decision(state, allowed)
        return self._query_llm_policy(state, allowed, view=view)

    def _log_tier2_io(self, state: dict[str, Any], decision: dict[str, Any], **fields: Any) -> dict[str, Any]:
        entry = {
            "generation": int(state.get("generation", self._generation_index())),
            "controller": self.controller,
            "query_index": int(self._llm_queries),
            "allowed_actions": list(fields.pop("allowed", ())),
            **fields,
            "controller_action": str(decision.get("action", "")),
            "final_action": str(decision.get("action", "")),
            "source": str(decision.get("source", "")),
            "reason_code": str(decision.get("reason_code", "")),
            "confidence": float(decision.get("confidence", 0.0)),
        }
        self.tier2_io_log.append(entry)
        self._pending_io = entry
        return entry

    def _random_decision(self, state: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
        self._last_llm_state = dict(state)
        rng = self._tier2_generator()
        action = allowed[int(rng.integers(len(allowed)))]
        self.llm_call_stats["accepted_actions"] += 1
        decision = {
            "action": action,
            "confidence": 1.0 / float(len(allowed)),
            "reason_code": "random_uniform",
            "source": "tier2",
        }
        self._log_tier2_io(state, decision, allowed=allowed)
        self._write_llm_usage_log(decision, state)
        return decision

    def _bandit_scores(self, allowed: Sequence[str]) -> dict[str, float]:
        """UCB1 over the EMA credit: ema[a] + c*sqrt(ln(1+t)/(1+n_a))."""
        t = float(self._bandit_t)
        bonus_num = float(_np.log1p(t))
        return {
            a: float(self._action_reward_ema.get(a, 0.0))
            + _BANDIT_UCB_C * float(_np.sqrt(bonus_num / (1.0 + float(self._bandit_counts.get(a, 0)))))
            for a in allowed
        }

    def _bandit_decision(self, state: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
        self._last_llm_state = dict(state)
        scores = self._bandit_scores(allowed)
        best = max(scores.values())
        ties = [a for a in allowed if scores[a] >= best - 1e-12]
        action = ties[0] if len(ties) == 1 else ties[int(self._tier2_generator().integers(len(ties)))]
        self._bandit_t += 1
        self._bandit_counts[action] = int(self._bandit_counts.get(action, 0)) + 1
        self.llm_call_stats["accepted_actions"] += 1
        decision = {
            "action": action,
            "confidence": 1.0 / float(len(ties)),
            "reason_code": "bandit_ucb1",
            "source": "tier2",
        }
        self._log_tier2_io(state, decision, allowed=allowed, scores=scores)
        self._write_llm_usage_log(decision, state)
        return decision

    def _build_blind_payload(self, view: dict[str, Any], allowed: tuple[str, ...]) -> tuple[dict[str, Any], str]:
        """Legacy prompt (pre-revision behaviour, byte-for-byte)."""
        narrative = self._state_to_narrative(view)
        history_shots = self._tier2_history(self._max_history_shots)
        payload = {
            "task": "select_nsga3_many_objective_policy_action",
            "state_narrative": narrative,
            "state": {k: view[k] for k in view if k != "allowed_actions"},
            "action_scores": {k: float(self._action_reward_ema.get(k, 0.0)) for k in allowed},
            "action_descriptions": {k: _LEGACY_ACTION_SEMANTICS.get(k, "") for k in allowed},
            "history": history_shots,
            "allowed_actions": list(allowed),
            "output_schema": dict(_OUTPUT_SCHEMA),
        }
        return payload, _LEGACY_SYSTEM_PROMPT

    def _build_informed_payload(self, view: dict[str, Any], allowed: tuple[str, ...]) -> tuple[dict[str, Any], str]:
        from core.llm.laya_policy import LEGACY_CHOICE_INSTRUCTIONS

        n_obj = int(view.get("n_obj", 0) or getattr(getattr(self, "problem", None), "n_obj", 5) or 5)
        payload = {
            "task": "select_nsga3_many_objective_policy_action",
            "mode": "laya_informed",
            "laya_text": self._informed_state_text(view, allowed),
            "laya_criteria": self._informed_criteria(allowed, n_obj),
            "laya_instructions": LEGACY_CHOICE_INSTRUCTIONS,
            "lang": "en",
            "allowed_actions": list(allowed),
            "output_schema": dict(_OUTPUT_SCHEMA),
        }
        return payload, _LEGACY_SYSTEM_PROMPT

    def _query_llm_policy(
        self,
        state: dict[str, Any],
        allowed: tuple[str, ...],
        view: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        view = state if view is None else view
        if self.controller == "laya_informed":
            prompt_payload, system_prompt = self._build_informed_payload(view, allowed)
        else:
            prompt_payload, system_prompt = self._build_blind_payload(view, allowed)
        prompt = json.dumps(prompt_payload, sort_keys=True)
        self._last_llm_state = dict(state)
        io: dict[str, Any] = {"client_prompt": prompt, "client_system": system_prompt}

        try:
            client = self._get_llm_client()
            try:
                request = build_laya_request(prompt, default_lang=str(getattr(client, "lang", "pt") or "pt"))
                io.update(
                    text=request["text"],
                    criteria=request["criteria"],
                    instructions=request["instructions"],
                    lang=request["lang"],
                )
            except Exception:  # logging must never change the decision path
                pass

            data = client.json_call(prompt=prompt, system=system_prompt)

            if not isinstance(data, dict) or "action" not in data:
                raise ValueError("Malformed LLM response JSON.")

            action = str(data.get("action", "")).strip().lower()
            confidence = max(0.0, min(float(data.get("confidence", 0.5)), 1.0))
            reason_code = str(data.get("reason_code", "llm_policy"))
            raw = {"choice": action, "confidence": confidence}
            if "raw_answer" in data:
                raw["model_answer"] = _jsonable(data.get("raw_answer"))
            io["raw_answer"] = raw

            if confidence < self.min_llm_confidence:
                self.llm_call_stats["below_confidence_floor"] += 1
                raise ValueError(f"Confidence {confidence:.2f} below threshold {self.min_llm_confidence:.2f}.")

            if action not in allowed:
                # Decision model suggested an action outside the current safe allowed set
                self.llm_call_stats["rejected_by_trap"] += 1
                fallback_action = max(allowed, key=lambda a: float(self._action_reward_ema.get(a, 0.0)))
                decision = {
                    "action": fallback_action,
                    "confidence": 0.3,
                    "reason_code": "trap_filtered_llm_fallback",
                    "source": "trap_fallback",
                    "_llm_suggested": action,
                }
                self._log_tier2_io(state, decision, allowed=allowed, **io)
                self._write_llm_usage_log(decision, state)
                return decision

            self.llm_call_stats["accepted_actions"] += 1
            decision = {
                "action": action,
                "confidence": confidence,
                "reason_code": reason_code,
                "source": "tier2",
            }
            self._log_tier2_io(state, decision, allowed=allowed, **io)
            self._write_llm_usage_log(decision, state)
            return decision

        except Exception as exc:
            # Graceful Fallback: Never crash or terminate the run.
            self.llm_call_stats["graceful_fallbacks"] += 1
            fallback_action = self._heuristic_action(view, allowed)
            decision = {
                "action": fallback_action,
                "confidence": 0.5,
                "reason_code": f"graceful_fallback_error: {type(exc).__name__}",
                "source": "error_fallback",
            }
            io.setdefault("raw_answer", None)
            io["error"] = f"{type(exc).__name__}: {exc}"
            self._log_tier2_io(state, decision, allowed=allowed, **io)
            self._write_llm_usage_log(decision, state)
            return decision

    def _write_llm_usage_log(self, decision: dict[str, Any], state: dict[str, Any]) -> None:
        if self.llm_log_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm": self.LOG_ALGORITHM_NAME,
            "controller": self.controller,
            "problem_name": self._problem_name(),
            "model": str(getattr(self.llm_client, "resolved_model", "none")),
            "generation": int(state.get("generation", self._generation_index())),
            "action": str(decision.get("action", "")),
            "confidence": float(decision.get("confidence", 0.0)),
            "reason_code": str(decision.get("reason_code", "")),
            "source": str(decision.get("source", "tier2")),
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
        # Refinement trap: stagnation on a well-covered, non-degenerate front (empty niches
        # moderate, many non-dominated solutions). Destructive exploration (var, div) is
        # suppressed so the run polishes the front instead of scattering it.
        if stagnation and empty_niches <= 0.40 and nd_ratio > 0.50:
            return ("conv", "pref", "subpop", "ref")

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

    def _consumed_fraction(self, state: dict[str, Any]) -> float:
        """Fraction of the evaluation budget consumed (true state, never masked)."""
        if self._resolve_n_max_evals() > 0:
            return float(max(0.0, min(1.0, 1.0 - float(state.get("budget_ratio", 1.0)))))
        est_generations = max(1.0, float(max(100, self.pop_size * 100)) / float(max(1, self.pop_size)))
        return float(max(0.0, min(1.0, float(state.get("generation", 0)) / est_generations)))

    def _fixed_schedule_due(self, state: dict[str, Any], budget: int) -> int:
        """Number of fixed-schedule queries due: #{k in 1..B : (k - 0.5)/B <= consumed}."""
        if budget <= 0:
            return 0
        consumed = self._consumed_fraction(state)
        return int(min(budget, max(0, int(_np.floor(consumed * budget + 0.5 + 1e-9)))))

    def _should_query_llm(self, state: dict[str, Any], allowed: tuple[str, ...]) -> bool:
        if not self.enable_llm or not allowed:
            return False
        if self.tier2_schedule == "fixed":
            budget = self._llm_call_budget()
            return self._llm_queries < min(budget, self._fixed_schedule_due(state, budget))
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

        sc = self.reward_scales
        angle_term = np.tanh(sc["angle"] * angle_disp_delta)
        improvement_term = np.tanh(sc["improvement"] * improvement)
        nd_term = np.tanh(sc["nd"] * nd_delta)
        entropy_term = np.tanh(sc["entropy"] * entropy_delta)
        w_imp, w_nd, w_ent, w_ang = self.reward_weights[action]
        reward = w_imp * improvement_term + w_nd * nd_term + w_ent * entropy_term + w_ang * angle_term

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

    # ------------------------------------------------------------------ prompt text
    def _state_to_narrative(self, state: dict[str, Any]) -> str:
        phase = str(state.get("phase", "middle"))
        budget = float(state.get("budget_ratio", 0.5))
        entropy = float(state.get("crowding_entropy", 0.5))
        nd_ratio = float(state.get("non_dominated_ratio", 0.3))
        empty = float(state.get("empty_niches", 0.2))
        stagnation = int(state.get("stagnation", 0)) > 0
        improvement = float(state.get("improvement", 0.0))
        theta = float(state.get("angle_dispersion", state.get("angular_dispersion", 0.5)))
        M = int(state.get("n_obj", 5))
        gen = int(state.get("generation", 0))

        entropy_label = "critically low" if entropy < 0.2 else "low" if entropy < 0.4 else "moderate" if entropy < 0.65 else "high"
        nd_label = "very high dominance ratio" if nd_ratio > 0.7 else "moderate" if nd_ratio > 0.3 else "sparse non-dominated front"
        stag_label = "active stagnation (no measurable improvement)" if stagnation else "no stagnation detected"
        budget_label = f"{budget*100:.0f}% budget remaining ({phase} phase)"
        empty_label = f"{empty*100:.0f}% of reference niches unoccupied"

        return (
            f"Many-objective NSGA-III search at generation {gen} for M={M} objectives with {budget_label}. "
            f"Crowding entropy is {entropy_label} ({entropy:.3f}). "
            f"Non-dominated ratio: {nd_ratio:.3f} ({nd_label}). "
            f"{empty_label}. "
            f"Recent improvement signal: {improvement:.6f}. "
            f"Angular mean dispersion: {theta:.3f}. "
            f"Population diagnostic: {stag_label}."
        )

    def _state_interpretation(self, view: dict[str, Any]) -> str:
        entropy = float(view.get("crowding_entropy", 0.5))
        nd_ratio = float(view.get("non_dominated_ratio", 0.5))
        empty = float(view.get("empty_niches", 0.2))
        stagnation = int(view.get("stagnation", 0)) > 0
        improvement = float(view.get("improvement", 0.0))
        phase = str(view.get("phase", "middle"))
        notes: list[str] = []
        if entropy < 0.20:
            notes.append("the population has collapsed onto very few reference niches")
        elif entropy < 0.45:
            notes.append("diversity across reference niches is low")
        else:
            notes.append("solutions are spread over many reference niches")
        if empty > 0.40:
            notes.append("most reference directions are unoccupied, so the front may be irregular, degenerate or disconnected")
        elif empty > 0.0:
            notes.append("some reference directions are unoccupied")
        if nd_ratio < 0.20:
            notes.append("most solutions are dominated, so convergence pressure is still useful")
        elif nd_ratio > 0.70:
            notes.append("almost all solutions are non-dominated, so dominance gives little selection pressure")
        if stagnation:
            notes.append("the best normalized objective sum has not improved over the stagnation window")
        elif improvement > 1e-4:
            notes.append("the search is still improving")
        notes.append(f"the run is in its {phase} phase")
        return "Interpretation: " + "; ".join(notes) + "."

    def _informed_state_text(self, view: dict[str, Any], allowed: tuple[str, ...]) -> str:
        narrative = self._state_to_narrative(view)
        numeric_keys = (
            "generation", "n_obj", "budget_ratio", "phase", "crowding_entropy", "non_dominated_ratio",
            "empty_niches", "angle_dispersion", "angle_dispersion_delta", "best_norm_sum", "improvement",
            "entropy_delta", "non_dominated_delta", "stagnation",
        )
        features: list[str] = []
        for key in numeric_keys:
            if key not in view:
                continue
            value = view[key]
            if isinstance(value, (bool, _np.bool_)) or key in ("generation", "n_obj", "stagnation"):
                features.append(f"{key}={int(value)}")
            elif isinstance(value, (int, float, _np.integer, _np.floating)):
                features.append(f"{key}={float(value):.4f}")
            else:
                features.append(f"{key}={value}")
        credit = "; ".join(f"{a}={float(self._action_reward_ema.get(a, 0.0)):+.4f}" for a in allowed)
        history = self._tier2_history(_INFORMED_HISTORY)
        if history:
            outcomes = "; ".join(
                f"gen {int(h.get('gen', 0))}: {h.get('action', '?')} -> improvement={float(h.get('improvement', 0.0)):+.5f}, "
                f"entropy_delta={float(h.get('entropy_delta', 0.0)):+.3f}, nd_delta={float(h.get('nd_delta', 0.0)):+.3f}"
                for h in history
            )
        else:
            outcomes = "none yet"
        n_obj = int(view.get("n_obj", 5) or 5)
        effects = "; ".join(
            "{}: pc={:g} eta_c={:g} pm={:g} eta_m={:g}".format(a, *(float(v) for v in self._variation_parameters(a, n_obj=n_obj)))
            for a in allowed
        )
        # Order matters: Laya truncates the state from the right, so the interpretation, the
        # action credit and recent outcomes come first and the raw numbers last.
        return (
            f"STATE: {self._state_interpretation(view)}\n"
            f"EMA CREDIT (higher is better): {credit}\n"
            f"LAST {_INFORMED_HISTORY} OUTCOMES (improvement > 0 is progress): {outcomes}\n"
            f"NARRATIVE: {narrative}\n"
            f"NUMERIC FEATURES: {'; '.join(features)}\n"
            f"OPERATOR SETTINGS: {effects}; var adds adaptive N(0, sigma*span) noise to 15% of variables "
            f"(sigma capped at 0.08, damped when converged); "
            f"ref keeps the axis directions fixed, returns empty directions {self.ref_gamma:.0%} toward their "
            f"initial position and moves occupied ones up to {self.ref_alpha_max:.0%} toward their niche centroid."
        )

    def _informed_criteria(self, allowed: Sequence[str], n_obj: int) -> dict[str, str]:
        """Cue-first option texts sized for Laya's head budget.

        Laya caps each option at 48 tokens and the whole head (instruction + options) at 192,
        shrinking every option to ~(192-16)/k tokens when k options overflow it; the question
        instruction itself is cut to the remaining budget (min 8 tokens). An earlier version put
        the "Use when" cue after a 55-105-token description, so the cue never reached the model
        (verifier round 3, 15/15 queries). Here the cue comes first and each option stays within
        _INFORMED_OPTION_MAX_TOKENS; the numeric operator effects travel in the state text
        (``_informed_state_text``), see tests/test_larc_f2.py::test_a4_informed_fits_laya_budget.
        """
        return {a: f"Use when {_USE_WHEN[a]} Effect: {_EFFECT_SHORT[a]}" for a in allowed if a in _USE_WHEN}

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

    # ------------------------------------------------------------------ operators
    def _mating_pool(self, rng: np.random.Generator) -> np.ndarray:
        """Zero-pressure mating, identical for every action: parents are drawn uniformly at random with replacement.

        With constraints, each parent is the winner of a binary tournament on the total constraint violation between
        two uniformly drawn candidates, a tie going to the first one; the pool is therefore exactly uniform whenever
        all solutions are feasible.  (A tournament on an all-zero fitness is not uniform: its ties go to the lower
        index, which favours the solutions stored first.)
        """
        n = len(self.pop)
        first = rng.integers(0, n, size=self.pop_size)
        cv = _constraint_violation(self.pop)
        if not np.any(cv > 0):
            return first
        second = rng.integers(0, n, size=self.pop_size)
        return np.where(cv[second] < cv[first], second, first)

    def _variation_parameters(self, action: str, n_obj: Optional[int] = None) -> list[float]:
        if n_obj is None:
            n_obj = int(getattr(getattr(self, "problem", None), "n_obj", 5) or 5)
        n_obj = int(n_obj)
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
        """Anchored reference direction adaptation on the unit simplex.

        Maintains strict confinement on Delta^{M-1} via convex combinations,
        relaxes inactive niches elastically toward initial Das-Dennis coordinates,
        tracks local niche centroids with density damping for active niches,
        and strictly preserves canonical extreme basis directions (A-NSGA-III).
        """
        F = _population_objectives(pop)
        norm = _normalized_objectives(F)
        # Private copy: never mutate an array that may be owned by the caller (A2).
        ref_dirs = _host_array(self.ref_dirs).copy()
        v0 = self._ref_dirs_initial
        if v0 is None:
            return

        niche, _ = _associate_ref_dirs(F, ref_dirs)
        niche_arr = _np.asarray(to_numpy(niche), dtype=int)
        n_ref = int(ref_dirs.shape[0])
        counts = _np.bincount(niche_arr, minlength=n_ref)

        extreme_mask = getattr(self, "_extreme_ref_mask", None)
        if extreme_mask is None or extreme_mask.shape[0] != n_ref:
            extreme_mask = _np.isclose(_np.max(v0, axis=1), 1.0) & _np.isclose(_np.sum(v0, axis=1), 1.0)

        # 1. Inactive intermediate niches: elastic canonical relaxation toward v0
        inactive = _np.where((counts == 0) & (~extreme_mask))[0]
        if inactive.size:
            gamma = self.ref_gamma
            ref_dirs[inactive] = (1.0 - gamma) * ref_dirs[inactive] + gamma * v0[inactive]

        # 2. Active intermediate niches: density-damped local centroid tracking with anchor restitution
        active = _np.where((counts > 0) & (~extreme_mask))[0]
        if active.size:
            norm_arr = _np.asarray(to_numpy(norm), dtype=float)
            norm_sum = _np.maximum(_np.sum(norm_arr, axis=1, keepdims=True), 1e-12)
            norm_simplex = norm_arr / norm_sum

            alpha_max = self.ref_alpha_max
            beta = self.ref_beta
            for idx in active:
                pts = norm_simplex[niche_arr == idx]
                if pts.shape[0] == 0:
                    continue
                local_c = _np.mean(pts, axis=0)
                local_c = local_c / max(float(_np.sum(local_c)), 1e-12)

                density = float(pts.shape[0])
                alpha = alpha_max * (density / (density + 1.0))

                ref_dirs[idx] = (1.0 - alpha - beta) * ref_dirs[idx] + alpha * local_c + beta * v0[idx]

        # 3. Canonical extreme basis directions remain strictly invariant
        ref_dirs[extreme_mask] = v0[extreme_mask]

        # 4. Convex closure and sanitization on unit simplex
        ref_dirs = _np.maximum(ref_dirs, 0.0)
        ref_dirs = ref_dirs / _np.maximum(_np.sum(ref_dirs, axis=1, keepdims=True), 1e-12)
        self.ref_dirs = _np.maximum(ref_dirs, 1e-12)
    def _record_ref_dirs_trajectory(self) -> None:
        if self._ref_dirs_initial is None:
            return
        current = _np.asarray(to_numpy(self.ref_dirs), dtype=float)
        if current.shape != self._ref_dirs_initial.shape:
            return
        dist = float(_np.sum(_np.abs(current - self._ref_dirs_initial)))
        self.ref_dirs_trajectory.append((int(self._generation_index()), dist))

    def _variable_classification_mutation(self, offspring: Population, rng: np.random.Generator) -> Population:
        X = np.asarray(offspring.get("X"), dtype=float)
        if X.size == 0:
            return offspring
        xl = np.asarray(self.problem.xl, dtype=float).reshape(-1)
        xu = np.asarray(self.problem.xu, dtype=float).reshape(-1)
        span = np.maximum(xu - xl, 1e-12)
        n_var = X.shape[1]
        n_mut = max(1, int(np.ceil(0.15 * n_var)))
        cols = rng.choice(n_var, size=n_mut, replace=False)
        # Option C (adaptive var): scale the decision-space kick by the current relative
        # spread of the decision variables. Early in the run (rel_std ~ std of U[0,1] ~ 0.29)
        # the kick keeps its full 0.08*span strength; once the population converges
        # (rel_std shrinks) the kick is damped to a minimum 10% so the operator stops
        # scattering a well-formed front. 0.25 approximates the uniform[0,1] std.
        base_sigma = float(getattr(self, "var_sigma_base", 0.08))
        pop_X = np.asarray(self.pop.get("X"), dtype=float) if getattr(self, "pop", None) is not None and len(self.pop) > 0 else X
        std = np.std(pop_X, axis=0, ddof=0) if pop_X.size else np.zeros(n_var)
        rel_std = np.clip(std[:n_var] / span[:n_var], 1e-3, 0.5)
        sigma_scale = np.clip(rel_std / 0.25, 0.10, 1.0)
        sigma_eff = base_sigma * sigma_scale
        noise = rng.normal(0.0, 1.0, size=(X.shape[0], n_mut)) * (sigma_eff[cols] * span[cols])[None, :]
        X_new = X.copy()
        X_new[:, cols] = np.clip(X_new[:, cols] + noise, xl[cols], xu[cols])
        return Population.new("X", X_new)

    def _inject_subpopulation_diversity(self, selected: Population, merged: Population) -> Population:
        """Swap best-angle candidates of empty niches in, keeping |pop| constant (A1).

        For every reference niche that is empty among the survivors but has at least one
        non-selected candidate in the merged (parent + offspring) pool -- processed in order
        of increasing candidate angle -- the candidate with the smallest angle to that
        direction replaces one survivor taken from the currently most crowded niche. Only
        niches with >= 2 members donate, so no swap empties another niche. Donor-niche ties
        are broken by the worst member; inside the donor niche the member removed is the one
        with the worst non-domination rank, ties broken by the largest angle (then index).
        """
        if selected is None or merged is None or len(selected) == 0 or len(merged) <= len(selected):
            return selected
        pos = {id(ind): i for i, ind in enumerate(merged)}
        sel_idx = [pos.get(id(ind)) for ind in selected]
        if any(i is None for i in sel_idx):
            return selected
        F = _population_objectives(merged)
        ref = _host_array(self.ref_dirs)
        niche, angle = _associate_ref_dirs(F, ref)
        niche = _np.asarray(to_numpy(niche), dtype=int)
        angle = _np.asarray(to_numpy(angle), dtype=float)
        n_ref = int(ref.shape[0])
        in_sel = _np.zeros(len(merged), dtype=bool)
        in_sel[_np.asarray(sel_idx, dtype=int)] = True
        counts = _np.bincount(niche[in_sel], minlength=n_ref)

        best_candidate: dict[int, int] = {}
        for r in _np.where(counts == 0)[0]:
            cands = _np.where((niche == r) & ~in_sel)[0]
            if cands.size:
                best_candidate[int(r)] = int(cands[int(_np.argmin(angle[cands]))])
        if not best_candidate:
            return selected

        try:
            front_no, _ = NDSort(F, _population_constraints(merged), len(merged))
            rank = _np.asarray(to_numpy(front_no), dtype=float).reshape(-1)
        except Exception:
            rank = _np.ones(len(merged), dtype=float)

        removed: list[int] = []
        injected: list[int] = []
        for r in sorted(best_candidate, key=lambda k: (angle[best_candidate[k]], k)):
            max_count = int(counts.max())
            if max_count < 2:
                break
            best_key = None
            victim = -1
            donor = -1
            for d in _np.where(counts == max_count)[0]:
                members = _np.where(in_sel & (niche == d))[0]
                worst = max(members.tolist(), key=lambda i: (rank[i], angle[i], i))
                key = (rank[worst], angle[worst], -int(d))
                if best_key is None or key > best_key:
                    best_key, victim, donor = key, int(worst), int(d)
            cand = best_candidate[r]
            in_sel[victim] = False
            in_sel[cand] = True
            counts[donor] -= 1
            counts[r] += 1
            removed.append(victim)
            injected.append(cand)
        if not injected:
            return selected
        self.llm_call_stats["subpop_swaps"] += len(injected)
        removed_set = set(removed)
        order = [i for i in sel_idx if i not in removed_set] + injected
        return merged[_np.asarray(order, dtype=int)]
