"""Compatibility facade: re-exports canonical MOEAD from algorithms.moead."""

from __future__ import annotations

from algorithms.moead.moead import *
from algorithms.moead import MOEAD, ALGORITHM_FLAGS

__all__ = ["MOEAD", "ALGORITHM_FLAGS"]
