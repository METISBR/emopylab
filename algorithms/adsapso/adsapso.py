# emopylab 2026
"""Adaptive dropout based surrogate-assisted particle swarm optimization.

Reference:
J. Lin, C. He, and R. Cheng. Adaptive dropout for high-dimensional expensive multiobjective optimization. Complex & Intelligent Systems, 2022, 8(1): 271C285.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'ADSAPSO': {'expensive', 'integer', 'many', 'multi', 'real'}}


class ADSAPSO(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
