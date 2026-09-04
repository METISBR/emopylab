# emopylab 2026
"""Grid-based evolutionary algorithm.\n\nReference:\nS. Yang, M. Li, X. Liu, and J. Zheng. A grid-based evolutionary algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2013, 17(5): 721-736.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'GrEA': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class GrEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
