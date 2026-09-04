# emopylab 2026
"""Constrained many-objective optimization with determinantal point processes.

Reference:
F. Ming, W. Gong, S. Li, L. Wang, and Z. Liao. Handling constrained many-objective optimization problems via determinantal point processes. Information Sciences, 2023, 643: 119260.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'CMaDPPs': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class CMaDPPs(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
