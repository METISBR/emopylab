# emopylab 2026
"""Hierarchy ranking based evolutionary algorithm.\n\nReference:\nW. Li, X. Yao, T. Zhang, R. Wang, and L. Wang. Hierarchy ranking method for multimodal multi-objective optimization with local Pareto fronts. IEEE Transactions on Evolutionary Computation, 2023, 27(1): 98-110.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'HREA': {'integer', 'multi', 'multimodal', 'real'}}


class HREA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
