# emopylab 2026
"""Multi-objective multifactorial evolutionary algorithm with subspace alignment and adaptive differential evolution.\n\nReference:\nZ. Liang, H. Dong, C. Liu, W. Liang, and Z. Zhu. Evolutionary multitasking for multiobjective optimization with subspace alignment and adaptive differential evolution. IEEE Transactions on Cybernetics, 2022, 52(4): 2096-2109.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MOMFEASADE': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MOMFEASADE(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
