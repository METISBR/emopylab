# emopylab 2026
"""Competitive and cooperative swarm optimization constrained multi-objective optimization algorithm.

Reference:
F. Ming, W. Gong, D. Li, L. Wang, and L. Gao. A competitive and cooperative swarm optimizer for constrained multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2023, 27(5): 1313-1326.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'CMOCSO': {'constrained', 'large', 'multi', 'real'}}


class CMOCSO(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
