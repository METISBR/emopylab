# emopylab 2026
"""APSEA (Adaptive population sizing evolutionary algorithm).

Reference:
Y. Tian, R. Wang, Y. Zhang, and X. Zhang. Adaptive population sizing for multi-population based constrained multi-objective optimization. Neurocomputing, 2025:129296.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'APSEA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class APSEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, alpha=0.05, beta=0.05, cp=5, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.cp = cp
