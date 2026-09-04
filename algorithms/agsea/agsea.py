# emopylab 2026
"""Automated guiding vector selection-based evolutionary algorithm.

Reference:
S. Shao, Y. Tian, and X. Zhang. Deep reinforcement learning assisted automated guiding vector selection for large-scale sparse multi-objective optimization. Swarm and Evolutionary Computation, 2024, 88: 101606.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'AGSEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class AGSEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
