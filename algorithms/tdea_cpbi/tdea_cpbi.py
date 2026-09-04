# emopylab 2026
"""Theta-dominance based evolutionary algorithm with CPBI.\n\nReference:\nF. Ming, W. Gong, L. Wang, and L. Gao. A constraint-handling technique for decomposition-based constrained many-objective evolutionary algorithms. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2023, 53(12): 7783-7793.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'tDEACPBI': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class tDEACPBI(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
