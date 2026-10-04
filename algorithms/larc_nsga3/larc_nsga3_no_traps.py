"""LARC-NSGA3-NoTraps: Ablation variant with unconstrained Tier-2 decisions (No Traps).

Allows the Tier-2 controller to select any action without being filtered by the
state-diagnostic safeguard traps (Degenerate-Front Trap and Multimodal Trap) or the
phase-dependent allowed sets (``enable_traps=False``). The ``controller`` argument is
passed through unchanged (the paper's ablation uses ``controller="laya_informed"``).
"""

from __future__ import annotations

from typing import Any
from .larc_nsga3 import LARC_NSGA3


class LARC_NSGA3_NoTraps(LARC_NSGA3):
    """LARC-NSGA3 without diagnostic safeguard traps (Unconstrained Tier 2)."""

    ALGO_FLAGS = {"multi", "many", "real", "integer"}
    OBJECTIVE_SCOPE = "many"
    LOG_ALGORITHM_NAME = "LARC_NSGA3_NoTraps"

    # explicit pop_size/ref_dirs: registries build algorithms from the constructor signature
    def __init__(self, pop_size: int = 100, ref_dirs: Any = None, **kwargs: Any) -> None:
        kwargs["enable_llm"] = True
        kwargs["enable_traps"] = False
        super().__init__(pop_size=pop_size, ref_dirs=ref_dirs, **kwargs)
