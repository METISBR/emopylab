"""LARC-NSGA3-NoTraps: Ablation variant with unconstrained LLM decisions (No Traps).

Allows the LLM to select any action without being filtered by the state-diagnostic
safeguard traps (Degenerate-Front Trap and Multimodal Trap).
"""

from __future__ import annotations

from typing import Any
from .larc_nsga3 import LARC_NSGA3


class LARC_NSGA3_NoTraps(LARC_NSGA3):
    """LARC-NSGA3 without diagnostic safeguard traps (Unconstrained Tier 2)."""

    ALGO_FLAGS = {"multi", "many", "real", "integer"}
    OBJECTIVE_SCOPE = "many"
    LOG_ALGORITHM_NAME = "LARC_NSGA3_NoTraps"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["enable_llm"] = True
        kwargs["enable_traps"] = False
        super().__init__(*args, **kwargs)
