# emopylab 2026
"""C3M (Constraint, multiobjective, multi-stage, multi-constraint EA).

Reference:
R. Sun, J. Zou, Y. Liu, S. Yang, and J. Zheng. A multi-stage algorithm for solving multi-objective optimization problems with multi-constraints. IEEE Transactions on Evolutionary Computation, 2023, 27(5): 1207-1219.
"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'C3M': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class C3M(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type
