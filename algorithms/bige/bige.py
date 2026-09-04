# emopylab 2026
"""BiGE (Bi-goal evolution).

Reference:
M. Li, S. Yang, and X. Liu. Bi-goal evolution for many-objective optimization problems. Artificial Intelligence, 2015, 228:45-65.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'BiGE': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class BiGE(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
