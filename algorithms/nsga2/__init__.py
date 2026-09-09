"""Official canonical NSGA-II package for EmoPyLab."""

from __future__ import annotations

from .nsga2 import NSGA2, binary_tournament

NSGAII = NSGA2
NSGA_II = NSGA2

ALGORITHMS = {
    "NSGA-II": NSGA2,
}

ALGORITHM_FLAGS = {
    "NSGA-II": {"multi", "real", "integer", "binary", "permutation", "label", "constrained"},
}

__all__ = [
    "NSGA2",
    "NSGAII",
    "NSGA_II",
    "binary_tournament",
    "ALGORITHMS",
    "ALGORITHM_FLAGS",
]
