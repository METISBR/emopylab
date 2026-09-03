"""Contextual Multi-Armed Bandit for MaACO operator selection.

Inspired by FRRMAB (Li et al., IEEE TEVC 2014) and extended to
many-objective optimization with LLM-observable state.

The bandit maintains an Upper Confidence Bound (UCB1) over the four
variation operator arms:
    - ``llm_sbx`` (Simulated Binary Crossover)
    - ``llm_de`` (Differential Evolution)
    - ``llm_perturb`` (Polynomial Mutation)
    - ``acor_mixture`` (Classical Continuous ACO kernel)

Reward Signal
-------------
At generation t, after environmental selection retains survivors,
the reward for the operator applied is the normalized improvement in
hypervolume:

    R_t = max(0, (HV_t - HV_{t-1}) / (HV_{t-1} + eps))

Arms with zero past selections are given priority (standard UCB warm-up).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class UCB1OperatorBandit:
    """UCB1 bandit for dynamic operator selection."""

    def __init__(self,
                 arm_names: Tuple[str, ...] = ("llm_sbx", "llm_de", "llm_perturb", "acor_mixture"),
                 c: float = 0.5,
                 window_size: int = 20,
                 decay: float = 0.95):
        self.arm_names = list(arm_names)
        self.n_arms = len(self.arm_names)
        self.c = float(c)  # exploration constant
        self.window_size = int(window_size)
        self.decay = float(decay)

        self.counts = {arm: 0 for arm in self.arm_names}
        self.rewards_history: Dict[str, List[float]] = {arm: [] for arm in self.arm_names}
        self.total_pulls = 0
        self.last_arm: Optional[str] = None

    def select(self, rng: np.random.Generator,
               llm_preference: Optional[str] = None) -> str:
        """Select the next operator.

        Selection hierarchy:
          1. Un-pulled arms (warm-up: pull each arm at least once).
          2. If the LLM expressed a strong preference and its chosen arm
             is competitive (UCB score >= 80% of the leader), accept it.
          3. Otherwise, pick the argmax of the UCB1 scores.
        """
        # 1. Warm-up
        for arm in self.arm_names:
            if self.counts[arm] == 0:
                self.last_arm = arm
                return arm

        # Compute UCB1 score per arm
        scores = {}
        for arm in self.arm_names:
            recent = self.rewards_history[arm][-self.window_size:]
            mean_r = float(np.mean(recent)) if recent else 0.0
            bonus = self.c * math.sqrt(2.0 * math.log(max(1, self.total_pulls)) / max(1, self.counts[arm]))
            scores[arm] = mean_r + bonus

        # 2. LLM preference gating
        if llm_preference in scores:
            best_score = max(scores.values())
            if scores[llm_preference] >= 0.8 * best_score:
                self.last_arm = llm_preference
                return llm_preference

        # 3. Argmax with random tiebreak
        best_arms = [arm for arm, s in scores.items() if math.isclose(s, max(scores.values()))]
        chosen = str(rng.choice(best_arms))
        self.last_arm = chosen
        return chosen

    def update(self, arm: str, reward: float) -> None:
        """Record the observed reward for the given arm."""
        if arm not in self.arm_names:
            return
        r = float(max(0.0, reward))
        self.counts[arm] += 1
        self.total_pulls += 1
        self.rewards_history[arm].append(r)
        # Cap history to prevent unbounded memory growth
        if len(self.rewards_history[arm]) > self.window_size * 4:
            self.rewards_history[arm] = self.rewards_history[arm][-self.window_size * 2:]

    def get_state(self) -> Dict[str, Any]:
        """Return a structured summary for the LLM prompt."""
        scores = {}
        means = {}
        for arm in self.arm_names:
            recent = self.rewards_history[arm][-self.window_size:]
            mean_r = float(np.mean(recent)) if recent else 0.0
            means[arm] = round(mean_r, 4)
            if self.counts[arm] > 0 and self.total_pulls > 0:
                bonus = self.c * math.sqrt(2.0 * math.log(self.total_pulls) / self.counts[arm])
                scores[arm] = round(mean_r + bonus, 4)
            else:
                scores[arm] = 999.0  # un-pulled
        return {
            "total_pulls": self.total_pulls,
            "counts": dict(self.counts),
            "recent_mean_rewards": means,
            "ucb_scores": scores,
            "last_arm": self.last_arm,
        }
