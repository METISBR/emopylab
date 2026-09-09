"""Official canonical RVEA package for EmoPyLab."""

from __future__ import annotations

from .rvea import RVEA

ALGORITHMS = {
    "RVEA": RVEA,
}

ALGORITHM_FLAGS = {
    "RVEA": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
}

__all__ = [
    "RVEA",
    "ALGORITHMS",
    "ALGORITHM_FLAGS",
]
