"""EmoPyLab Multi-Objective Optimization (MOO) Algorithm Catalog."""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2, binary_tournament
from algorithms.moo.nsga3 import NSGA3, ReferenceDirectionSurvival, associate_to_niches
from algorithms.moo.moead import MOEAD
from algorithms.moo.rvea import RVEA
from algorithms.moo.age2 import AGEMOEA2
from algorithms.moo.sms import SMSEMOA, cv_and_dom_tournament

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
