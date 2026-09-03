# emopylab 2026
"""Indicator-based constrained multi-objective algorithm.\n\nReference:\nJ. Yuan, H. Liu, Y. Ong, and Z. He. Indicator-based evolutionary algorithm for solving constrained multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2022, 26(2): 379-391.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'ICMA': {'constrained', 'integer', 'multi', 'real'}}


class ICMA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
