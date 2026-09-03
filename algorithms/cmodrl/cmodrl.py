# emopylab 2026
"""Constrained multiobjective optimization via deep reinforcement learning.

Reference:
F. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin. Automated configuration of evolutionary algorithms via deep reinforcement learning for constrained multiobjective optimization. IEEE Transactions on Cybernetics, 2025, 55(12): 5877-5890.
"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CMODRL': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CMODRL(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type
