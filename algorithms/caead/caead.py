# emopylab 2026
"""CAEAD (Dual-population algorithm based on alternative evolution and degeneration).

Reference:
J. Zou, R. Sun, S. Yang, and J. Zheng. A dual-population algorithm based on alternative evolution and degeneration for solving constrained multi-objective optimization problems. Information Sciences, 2021, 239: 89-102.
"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CAEAD': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CAEAD(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type
