# emopylab 2026
"""Vector angle based evolutionary algorithm.\n\nReference:\nY. Xiang, Y. Zhou, M. Li, and Z. Chen. A vector angle-based evolutionary algorithm for unconstrained many-objective optimization. IEEE Transactions on Evolutionary Computation, 2017, 21(1): 131-152.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'VaEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class VaEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
