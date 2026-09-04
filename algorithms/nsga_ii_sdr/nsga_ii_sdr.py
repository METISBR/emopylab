# emopylab 2026
"""NSGA-II with strengthened dominance relation.\n\nReference:\nY. Tian, R. Cheng, X. Zhang, Y. Su, and Y. Jin. A strengthened dominance relation considering convergence and diversity for evolutionary many- objective optimization. IEEE Transactions on Evolutionary Computation, 2019, 23(2): 331-345.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NSGAIISDR': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class NSGAIISDR(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
