"""Compatibility facade: re-exports canonical RVEA from algorithms.rvea."""

from __future__ import annotations

from algorithms.rvea.rvea import *
from algorithms.rvea import RVEA, ALGORITHM_FLAGS

__all__ = ["RVEA", "ALGORITHM_FLAGS"]
