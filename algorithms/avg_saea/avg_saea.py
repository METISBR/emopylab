# emopylab 2026
"""Adaptive variable grouping based surrogate-assisted evolutionary algorithm.

Reference:
Y. Li, X. Feng, and H. Yu. Solving high-dimensional expensive multiobjective optimization problems by adaptive decision variable grouping. IEEE Transactions on Evolutionary Computation, 2025, 29(4): 1041-1054.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'AVGSAEA': {'expensive', 'integer', 'large', 'multi', 'real'}}


class AVGSAEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
