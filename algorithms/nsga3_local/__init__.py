"""Backward compatibility shim package for legacy algorithms.nsga3_local imports.

Canonical implementation lives in algorithms.nsga3.
"""

from __future__ import annotations

from algorithms.nsga3.nsga3 import (
    NSGA3,
    NSGA3 as NSGA3Local,
    NSGA3 as NSGAIII,
    HyperplaneNormalization,
    _environmental_selection,
    _last_selection,
    _perpendicular_distance,
    _fronts,
    _constraint_violation,
    _population_objectives,
    _population_constraints,
    _update_zmin,
    ALGORITHMS,
    ALGORITHM_FLAGS,
)

__all__ = [
    "NSGA3",
    "NSGA3Local",
    "NSGAIII",
    "HyperplaneNormalization",
    "_environmental_selection",
    "_last_selection",
    "_perpendicular_distance",
    "_fronts",
    "_constraint_violation",
    "_population_objectives",
    "_population_constraints",
    "_update_zmin",
    "ALGORITHMS",
    "ALGORITHM_FLAGS",
]
