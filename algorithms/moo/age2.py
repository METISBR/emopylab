"""Compatibility facade: re-exports canonical AGEMOEA2 from algorithms.age2."""

from __future__ import annotations

from algorithms.age2.age2 import *
from algorithms.age2 import AGEMOEA2, ALGORITHM_FLAGS

__all__ = ["AGEMOEA2", "ALGORITHM_FLAGS"]
