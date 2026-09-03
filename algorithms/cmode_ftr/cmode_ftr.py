# emopylab 2026
"""CMODE-FTR (Constrained MODE based on fusion of two rankings).

Reference:
Z. Zeng, X. Zhang, and Z. Hong. A constrained multiobjective differential evolution algorithm based on the fusion of two rankings. Information Sciences, 2023, 647:119572.
"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'CMODEFTR': {'constrained', 'integer', 'multi', 'real'}}


class CMODEFTR(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, ranking_fusion=0.5, **kwargs):
        super().__init__(pop_size=pop_size, **kwargs)
        self.ranking_fusion = ranking_fusion
