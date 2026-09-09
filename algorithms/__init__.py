"""Algorithms package for EmoPyLab.

Official canonical multi-objective algorithms:
  - NSGA2 / NSGAII: Canonical NSGA-II (algorithms.nsga2)
  - NSGA3 / NSGAIII: Canonical NSGA-III (algorithms.nsga3)
  - RVEA: Canonical Reference Vector Guided Evolutionary Algorithm (algorithms.rvea)
  - MOEAD: Canonical MOEA/D (algorithms.moead)
  - AGEMOEA2: Canonical AGE-MOEA-II (algorithms.age2)
  - SMSEMOA: Canonical SMS-EMOA (algorithms.sms)

Specialized research variants reside in their dedicated isolated subpackages.
"""

from __future__ import annotations

from algorithms.nsga2 import NSGA2, NSGAII
from algorithms.nsga3 import NSGA3, NSGAIII
from algorithms.rvea import RVEA
from algorithms.moead import MOEAD
from algorithms.age2 import AGEMOEA2
from algorithms.sms import SMSEMOA

__all__ = [
    "NSGA2",
    "NSGAII",
    "NSGA3",
    "NSGAIII",
    "RVEA",
    "MOEAD",
    "AGEMOEA2",
    "SMSEMOA",
]
