# emopylab 2026
"""Constrained multiobjective evolutionary algorithm with multiple stages.

Reference:
Y. Tian, Y. Zhang, Y. Su, X. Zhang, K. C. Tan, and Y. Jin. Balancing objective optimization and constraint satisfaction in constrained evolutionary multi-objective optimization. IEEE Transactions on Cybernetics, 2022, 52(9): 9559-9572.
"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CMOEAMS': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CMOEAMS(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type
