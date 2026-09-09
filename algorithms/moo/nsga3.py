"""Compatibility facade: re-exports canonical NSGA3 from algorithms.nsga3."""

from __future__ import annotations

from algorithms.nsga3.nsga3 import *
from algorithms.nsga3 import NSGA3, NSGAIII, NSGA3Local, ALGORITHM_FLAGS

__all__ = ["NSGA3", "NSGAIII", "NSGA3Local", "ALGORITHM_FLAGS"]
