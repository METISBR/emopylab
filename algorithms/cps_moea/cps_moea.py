# emopylab 2026
"""CPS-MOEA (Classification and Pareto domination based MOEA).

Reference:
J. Zhang, A. Zhou, and G. Zhang. A classification and Pareto domination based multiobjective evolutionary algorithm. CEC 2015, 2883-2890.
"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'CPSMOEA': {'expensive', 'integer', 'multi', 'real'}}


class CPSMOEA(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, M_offspring=3, **kwargs):
        super().__init__(pop_size=pop_size, **kwargs)
        self.M_offspring = M_offspring
