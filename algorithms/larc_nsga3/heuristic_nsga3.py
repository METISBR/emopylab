"""Heuristic-NSGA3: Standalone Tier-1 Heuristic baseline for LARC ablation.

Executes the exact same adaptive operators and state-aware decision tree as LARC-NSGA3,
but runs 100% deterministically without any Tier-2 query (``controller="heuristic"``,
B = 0); no decision model is constructed or loaded.
"""

from __future__ import annotations

from typing import Any
from .larc_nsga3 import LARC_NSGA3


class Heuristic_NSGA3(LARC_NSGA3):
    """Standalone Heuristic-NSGA3 baseline (Tier 1 Only)."""

    ALGO_FLAGS = {"multi", "many", "real", "integer"}
    OBJECTIVE_SCOPE = "many"
    LOG_ALGORITHM_NAME = "Heuristic_NSGA3"

    # explicit pop_size/ref_dirs: registries build algorithms from the constructor signature
    def __init__(self, pop_size: int = 100, ref_dirs: Any = None, **kwargs: Any) -> None:
        kwargs["controller"] = "heuristic"
        kwargs["enable_llm"] = False
        kwargs["enable_traps"] = True
        super().__init__(pop_size=pop_size, ref_dirs=ref_dirs, **kwargs)
