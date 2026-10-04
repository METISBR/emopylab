# emopylab 2026
"""PEAplus (Kriging-assisted constrained EA with probabilistic dominance)."""

from __future__ import annotations

from algorithms.community_utils.pea import PEABase

ALGORITHM_FLAGS = {'PEAplus': {'constrained', 'expensive', 'integer', 'multi', 'real'}}


class PEAplus(PEABase):
    PLUS = True
