# emopylab 2026
"""GCS-MaOEA: Geometry-Coupled Search for Many-Objective Optimization.

A single online-learned reference geometry couples the search in objective space
(adaptive reference-vector repair) and decision space (direction-partitioned
variable analysis), driving direction-targeted reproduction.
"""
from __future__ import annotations

from .gcs_maoea import GCSMaOEA

__all__ = ["GCSMaOEA"]
