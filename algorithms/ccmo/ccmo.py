# emopylab 2026
"""Coevolutionary constrained multi-objective optimization framework.

Reference:
Y. Tian, T. Zhang, J. Xiao, X. Zhang, and Y. Jin. A coevolutionary framework for constrained multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2021, 25(1): 102-116.
"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CCMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type
