# emopylab 2026
"""Constrained two-stage evolutionary algorithm.

Reference:
F. Ming, W. Gong, H. Zhen, S. Li, L. Wang, and Z. Liao. A simple two-stage evolutionary algorithm for constrained multi-objective optimization. Knowledge-Based Systems, 2021, 228: 107263.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'CTSEA': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class CTSEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
