# emopylab 2026
"""Steady-state and generational evolutionary algorithm.\n\nReference:\nS. Jiang and S. Yang. A steady-state and generational evolutionary algorithm for dynamic multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2017, 21(1): 65-82.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'SGEA': {'binary', 'constrained', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


class SGEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
