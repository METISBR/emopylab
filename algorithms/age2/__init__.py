"""Official canonical AGE-MOEA-II package for EmoPyLab."""

from __future__ import annotations

from .age2 import AGEMOEA2

AGE2 = AGEMOEA2
AGE_MOEA2 = AGEMOEA2

ALGORITHMS = {
    "AGEMOEA2": AGEMOEA2,
    "AGE-MOEA-II": AGEMOEA2,
}

ALGORITHM_FLAGS = {
    "AGEMOEA2": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
    "AGE-MOEA-II": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
}

__all__ = [
    "AGEMOEA2",
    "AGE2",
    "AGE_MOEA2",
    "ALGORITHMS",
    "ALGORITHM_FLAGS",
]
