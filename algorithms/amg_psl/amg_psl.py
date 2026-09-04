# emopylab 2026
"""Adaptive multi-granular Pareto-optimal subspace learning.

Reference:
C. Sun, Y. Tian, S. Shao, S. Yang, and X. Zhang. An adaptive multi- granular Pareto-optimal subspace learning algorithm for sparse large- scale multi-objective optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2025.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'AMGPSL': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class AMGPSL(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
