# emopylab 2026
"""Constrained multi-objective optimization based on evolutionary multitasking optimization.\n\nReference:\nF. Ming, W. Gong, L. Wang, and L. Gao. Constrained multi-objective optimization via multitasking and knowledge transfer. IEEE Transactions on Evolutionary Computation, 2024, 28(1): 77-89.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CMOEMT': {'constrained', 'multi', 'real'}}


class CMOEMT(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
