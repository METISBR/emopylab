# emopylab 2026
"""Constrained many-objective evolutionary algorithm with enhanced mating and environmental selections.

Reference:
F. Ming, W. Gong, L. Wang, and L. Gao. A constrained many-objective optimization evolutionary algorithm with enhanced mating and environmental selections. IEEE Transactions on Cybernetics, 2023, 53(8): 4934-4946.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'CMME': {'binary', 'constrained', 'integer', 'label', 'many', 'permutation', 'real'}}


class CMME(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
