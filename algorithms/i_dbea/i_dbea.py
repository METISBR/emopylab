# emopylab 2026
"""Improved decomposition-based evolutionary algorithm.\n\nReference:\nM. Asafuddoula, T. Ray, and R. Sarker. A decomposition-based evolutionary algorithm for many objective optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(3): 445-460.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'IDBEA': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class IDBEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
