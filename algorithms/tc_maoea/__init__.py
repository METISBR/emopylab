# emopylab 2026
"""TC-MaOEA: Tangent-Coupled Many-Objective Evolutionary Algorithm.

Couples objective-space manifold SVD tangent bundles with decision-space
pullback projectors and KKT Pareto-stationarity descent for degenerate MaOPs.
"""
from __future__ import annotations

from .tc_maoea import TCMaOEA

__all__ = ["TCMaOEA"]
