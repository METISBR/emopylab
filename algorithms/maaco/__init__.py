"""MaACO v2 — Many-Objective Ant Colony Optimization.

Released alongside the ACDSA 2027 paper:
    "MaACO: An Angle-Penalized Ant Colony Optimization Algorithm
     with LLM-Guided Adaptation for Many-Objective Problems"
    by Thiago Santos and Sebastião Xavier (UFOP, METISBr).

Exports
-------
- ``MaACO`` — main algorithm class (EmoPyLab-compatible).
- ``llm_sbx``, ``llm_de``, ``llm_perturb``, ``acor_mixture`` — variation operators.
- ``UCB1OperatorBandit`` — contextual bandit for dynamic operator selection.
"""

from .bandit import UCB1OperatorBandit
from .maaco import ALGORITHM_FLAGS, MaACO
from .operators import (
    OPERATOR_NAMES,
    acor_mixture,
    apply_operator,
    llm_de,
    llm_perturb,
    llm_sbx,
)

__all__ = [
    "ALGORITHM_FLAGS",
    "MaACO",
    "OPERATOR_NAMES",
    "UCB1OperatorBandit",
    "acor_mixture",
    "apply_operator",
    "llm_de",
    "llm_perturb",
    "llm_sbx",
]
