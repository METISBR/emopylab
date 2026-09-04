# emopylab 2026
"""Improved SparseEA.\n\nReference:\nY. Zhang, Y. Tian, and X. Zhang. Improved SparseEA for sparse large-scale multi-objective optimization problems. Complex & Intelligent Systems, 2023, 9: 1127-1142.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SparseEA2': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class SparseEA2(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
