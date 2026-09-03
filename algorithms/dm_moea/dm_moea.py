# emopylab 2026
"""Dual model based multi-objective evolutionary algorithm.\n\nReference:\nP. Zhang, R. Zhang, Y. Tian, K. C. Tan, and X. Zhang. A dual model-based evolutionary framework for dynamic large-scale sparse multiobjective optimization. Swarm and Evolutionary Computation, 2025, 97: 102011.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'DMMOEA': {'binary', 'constrained', 'dynamic', 'integer', 'large', 'multi', 'real', 'sparse'}}


class DMMOEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
