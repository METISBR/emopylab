# emopylab 2026
"""Promising-region based EMO algorithm.\n\nReference:\nJ. Yuan, H. Liu, F. Gu, Q. Zhang, and Z. He. Investigating the properties of indicators and an evolutionary many-objective algorithm based on a promising region. IEEE Transactions on Evolutionary Computation, 2021, 25(1): 75-86.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'PREA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class PREA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
