# emopylab 2026
"""AR-MOEA (Adaptive reference-point based MOEA).

Reference:
Y. Tian, R. Cheng, X. Zhang, and Y. Jin. An indicator-based multiobjective evolutionary algorithm with reference point adaptation for better versatility. IEEE Transactions on Evolutionary Computation, 2018, 22(4): 609-622.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'ARMOEA': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class ARMOEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
