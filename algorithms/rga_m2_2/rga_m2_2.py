# emopylab 2026
"""RGA_M2_2 (Kriging-assisted GA with constraint surrogates)."""

from __future__ import annotations

from algorithms.community_utils.rga import RGABase

ALGORITHM_FLAGS = {'RGA_M2_2': {'constrained', 'expensive', 'multi', 'real'}}


class RGA_M2_2(RGABase):
    CV_MODE = 2
