# emopylab 2026
"""Kriging-assisted evolutionary algorithm with two search modes.\n\nReference:\nZ. Song, H. Wang, B. Xue, M. Zhang, and Y. Jin. Balancing objective optimization and constraint satisfaction in expensive constrained evolutionary multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2024, 28(5): 1286-1300.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'KTS': {'constrained', 'expensive', 'many', 'multi', 'real'}}


class KTS(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
