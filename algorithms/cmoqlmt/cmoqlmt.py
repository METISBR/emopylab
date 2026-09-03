# emopylab 2026
"""Constrained multi-objective optimization based on Q-learning and multitasking.\n\nReference:\nF. Ming, W. Gong, and L. Gao. Adaptive auxiliary task selection for multitasking-assisted constrained multi-objective optimization [feature]. IEEE Computational Intelligence Magazine, 2023, 18(2): 18-30.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CMOQLMT': {'constrained', 'multi', 'real'}}


class CMOQLMT(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
