# emopylab 2026
"""BL-SAEA (Bi-level surrogate modelling EA).

Reference:
H. Jiang, K. Qiu, Y. Tian, X. Zhang, and Y. Jin. Efficient surrogate modeling method for evolutionary algorithm to solve bilevel optimization problems. IEEE Transactions on Cybernetics, 2024, 54(7): 4335-4347.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'BLSAEA': {'bilevel', 'multi', 'real'}}


class BLSAEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
