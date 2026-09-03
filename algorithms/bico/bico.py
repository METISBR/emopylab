# emopylab 2026
"""BiCo (Bidirectional coevolution constrained MOEA).

Reference:
Z. Liu, B. Wang, and K. Tang. Handling constrained multiobjective optimization problems via bidirectional coevolution. IEEE Transactions on Cybernetics, 2022, 52(10): 10163-10176.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'BiCo': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class BiCo(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
