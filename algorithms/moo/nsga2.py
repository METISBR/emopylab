"""Compatibility facade: re-exports canonical NSGA2 from algorithms.nsga2."""

from __future__ import annotations

from algorithms.nsga2.nsga2 import *
from algorithms.nsga2 import NSGA2, binary_tournament, ALGORITHM_FLAGS

__all__ = ["NSGA2", "binary_tournament", "ALGORITHM_FLAGS"]
