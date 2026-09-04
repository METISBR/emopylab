# emopylab 2026
"""MOEA/D with detect-and-escape strategy.\n\nReference:\nQ. Zhu, Q. Zhang, and Q. Lin. A constrained multi-objective evolutionary algorithm with detect-and-escape strategy. IEEE Transactions on Evolutionary Computation, 2020, 24(5): 938-947.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MOEADDAE': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MOEADDAE(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
