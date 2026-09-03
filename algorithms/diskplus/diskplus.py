# emopylab 2026
"""Distribution-based Kriging-assisted constrained evolutionary algorithm.\n\nReference:\nZ. Zhang, Y. Wang, G. Sun, and T. Pang. A distribution information based Kriging-assisted evolutionary algorithm for expensive many-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2656-2670.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DISKplus': {'constrained', 'expensive', 'integer', 'many', 'multi', 'real'}}


class DISKplus(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
