"""EmoPyLab Multi-Objective Optimization (MOO) Algorithm Catalog (Canonical Facade)."""

from __future__ import annotations

from algorithms.nsga2 import NSGA2, binary_tournament
from algorithms.nsga3 import NSGA3, ReferenceDirectionSurvival, associate_to_niches
from algorithms.moead import MOEAD
from algorithms.rvea import RVEA
from algorithms.age2 import AGEMOEA2
from algorithms.sms import SMSEMOA, cv_and_dom_tournament

__all__ = [
    "NSGA2",
    "binary_tournament",
    "NSGA3",
    "ReferenceDirectionSurvival",
    "associate_to_niches",
    "MOEAD",
    "RVEA",
    "AGEMOEA2",
    "SMSEMOA",
    "cv_and_dom_tournament",
]
