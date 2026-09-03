# emopylab 2026
"""Multiobjective optimization framework based on nondominated sorting and.\n\nReference:\nB. Chen, W. Zeng, Y. Lin, and D. Zhang. A new local search-based multiobjective optimization algorithm. IEEE Transactions on Evolutionary Computation, 2015, 19(1): 50-73.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NSLS': {'integer', 'multi', 'real'}}


class NSLS(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
