# emopylab 2026
"""Adaptive merging and coordinated offspring generation based multi-modal multi-objective evolutionary algorithm.

Reference:
X. Wang, T. Zheng, and Y. Jin. Adaptive merging and coordinated offspring generation in multi-population evolutionary multi-modal multi-objective optimization. Proceedings of the International Conference on Data-driven Optimization of Complex Systems, 2023.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'ACMMEA': {'integer', 'large', 'multi', 'multimodal', 'real', 'sparse'}}


class ACMMEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
