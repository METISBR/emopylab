"""Official canonical NSGA-III package for EmoPyLab."""

from __future__ import annotations

from .nsga3 import NSGA3, ReferenceDirectionSurvival, associate_to_niches

NSGAIII = NSGA3
NSGA_III = NSGA3

ALGORITHMS = {
    "NSGA3": NSGA3,
    "NSGA-III": NSGA3,
}

ALGORITHM_FLAGS = {
    "NSGA3": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
    "NSGA-III": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
}

__all__ = [
    "NSGA3",
    "NSGAIII",
    "NSGA_III",
    "ReferenceDirectionSurvival",
    "associate_to_niches",
    "ALGORITHMS",
    "ALGORITHM_FLAGS",
]
