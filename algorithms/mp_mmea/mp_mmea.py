# emopylab 2026
"""Multi-population multi-modal multi-objective evolutionary algorithm.\n\nReference:\nY. Tian, R. Liu, X. Zhang, H. Ma, K. C. Tan, and Y. Jin. A multipopulation evolutionary algorithm for solving large-scale multimodal multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2021, 25(3): 405-418.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MPMMEA': {'integer', 'large', 'multi', 'multimodal', 'real', 'sparse'}}


class MPMMEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
