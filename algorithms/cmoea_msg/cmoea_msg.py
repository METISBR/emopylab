# emopylab 2026
"""Multi-stage constrained multi-objective evolutionary algorithm.

Reference:
Y. Tian, J. Chen, and X. Zhang. An optimizer combining evolutionary computation and gradient descent for constrained multi-objective optimization. Journal of Computer Applications (Chinese), 2024, 44(05): 1386-1392.
"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CMOEAMSG': {'constrained', 'integer', 'multi', 'real'}}


class CMOEAMSG(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type
