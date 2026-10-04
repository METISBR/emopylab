# emopylab 2026
"""DISK (Kriging-assisted EA with density-weighted probabilistic dominance and reference-vector local search)."""

from __future__ import annotations

from algorithms.community_utils.disk import DISKBase

ALGORITHM_FLAGS = {'DISK': {'expensive', 'integer', 'many', 'multi', 'real'}}


class DISK(DISKBase):
    PLUS = False
