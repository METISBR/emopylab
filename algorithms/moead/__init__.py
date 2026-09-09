"""Official canonical MOEA/D package for EmoPyLab."""

from __future__ import annotations

from .moead import MOEAD

ALGORITHMS = {
    "MOEAD": MOEAD,
    "MOEA/D": MOEAD,
}

ALGORITHM_FLAGS = {
    "MOEAD": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
    "MOEA/D": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
}

__all__ = [
    "MOEAD",
    "ALGORITHMS",
    "ALGORITHM_FLAGS",
]
