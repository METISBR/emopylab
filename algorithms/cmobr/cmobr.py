# emopylab 2026
"""Constrained multiobjective optimization via both constraint and objective relaxations.

Reference:
F. Ming, B. Xue, M. Zhang, W. Gong, and H. Zhen. Constrained multiobjective optimization via relaxations on both constraints and objectives. IEEE Transactions on Artificial Intelligence, 2024, 5(12): 6709-6722.
"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'CMOBR': {'constrained', 'integer', 'many', 'multi', 'real'}}


class CMOBR(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
