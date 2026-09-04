# emopylab 2026
"""Stochastic ranking algorithm.\n\nReference:\nB. Li, K. Tang, J. Li, and X. Yao. Stochastic ranking algorithm for many-objective optimization based on multiple indicators. IEEE Transactions on Evolutionary Computation, 2016, 20(6): 924-938.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SRA': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class SRA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
