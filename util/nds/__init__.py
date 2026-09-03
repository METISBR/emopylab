"""EmoPyLab Non-Dominated Sorting Package (zero-pymoo standalone)."""

from __future__ import annotations

from util.nds.non_dominated_sorting import (
    NonDominatedSorting,
    fast_non_dominated_sort,
    find_non_dominated,
)

__all__ = [
    "NonDominatedSorting",
    "fast_non_dominated_sort",
    "find_non_dominated",
]
