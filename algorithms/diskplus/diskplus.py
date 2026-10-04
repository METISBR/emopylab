# emopylab 2026
"""DISKplus (Kriging-assisted EA with density-weighted probabilistic dominance and reference-vector local search)."""

from __future__ import annotations

from algorithms.community_utils.disk import DISKBase

ALGORITHM_FLAGS = {'DISKplus': {'constrained', 'expensive', 'integer', 'many', 'multi', 'real'}}


class DISKplus(DISKBase):
    PLUS = True
