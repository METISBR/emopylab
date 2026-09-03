# emopylab 2026
"""Multiform optimization framework based on SPEA2.\n\nReference:\nR. Jiao, B. Xue, and M. Zhang. A multiform optimization framework for constrained multiobjective optimization. IEEE Transactions on Cybernetics, 2023, 53(8):5165-5177.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MFOSPEA2': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MFOSPEA2(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
