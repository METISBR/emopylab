# emopylab 2026
"""Duplication analysis based evolutionary algorithm.\n\nReference:\nH. Xu, B. Xue, and M. Zhang. A duplication analysis based evolutionary algorithm for bi-objective feature selection. IEEE Transactions on Evolutionary Computation, 2021, 25(2): 205-218.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'DAEA': {'binary', 'multi'}}


class DAEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
