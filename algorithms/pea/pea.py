# emopylab 2026
"""PEA (Kriging-assisted constrained EA with probabilistic dominance)."""

from __future__ import annotations

from algorithms.community_utils.pea import PEABase

ALGORITHM_FLAGS = {'PEA': {'constrained', 'expensive', 'integer', 'multi', 'real'}}


class PEA(PEABase):
    PLUS = False
