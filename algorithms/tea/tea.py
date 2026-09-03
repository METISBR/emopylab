# emopylab 2026
"""Two-phase evolutionary algorithm.\n\nReference:\nZ. Zhang, Y. Wang, J. Liu, G. Sun, and K. Tang. A two-phase Kriging- assisted evolutionary algorithm for expensive constrained multiobjective optimization problems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(8): 4579-4591.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'TEA': {'constrained', 'expensive', 'integer', 'many', 'multi', 'real'}}


class TEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
