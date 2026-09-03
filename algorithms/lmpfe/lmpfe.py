# emopylab 2026
"""Evolutionary algorithm with local model based Pareto front estimation.\n\nReference:\nY. Tian, L. Si, X. Zhang, K. C. Tan, and Y. Jin. Local model based Pareto front estimation for multi-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2023, 53(1): 623-634.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'LMPFE': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class LMPFE(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
