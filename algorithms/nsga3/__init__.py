"""Official canonical NSGA-III package for EmoPyLab."""

from __future__ import annotations

from .nsga3 import (
    NSGA3,
    NSGAIII,
    ReferenceDirectionSurvival,
    associate_to_niches,
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
    "ReferenceDirectionSurvival",
    "associate_to_niches",
    "ALGORITHM_FLAGS",
]
