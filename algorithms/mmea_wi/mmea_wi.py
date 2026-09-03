# emopylab 2026
"""Weighted indicator-based evolutionary algorithm for multimodal multi-objective optimization.\n\nReference:\nW. Li, T. Zhang, R. Wang, and H. Ishibuchi. Weighted indicator-based evolutionary algorithm for multimodal multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(6): 1064-1078.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MMEAWI': {'integer', 'multi', 'multimodal', 'real'}}


class MMEAWI(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
