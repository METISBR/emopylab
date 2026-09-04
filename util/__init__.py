"""Local compatibility shims for legacy util imports (emopylab 2026)."""

from __future__ import annotations

from util.misc import (
    at_least_2d_array,
    cdist,
    crossover_mask,
    default_random_state,
    find_duplicates,
    get_duplicates,
    has_feasible,
    powerset,
    row_at_least_once_true,
)

__all__ = [
    "default_random_state",
    "at_least_2d_array",
    "cdist",
    "crossover_mask",
    "row_at_least_once_true",
    "find_duplicates",
    "get_duplicates",
    "has_feasible",
    "powerset",
]
