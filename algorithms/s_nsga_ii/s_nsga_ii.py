# emopylab 2026
"""Sparse nondominated sorting genetic algorithm II.\n\nReference:\nI. Kropp, A. Pouyan Nejadhashemi, and K. Deb. Improved evolutionary operators for sparse large-scale multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2024, 28(2): 460-473. -------------------------------------------------------------------------- This function is written by Ian Meyer Kropp Generate random population Optimization\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SNSGAII': {'constrained', 'large', 'multi', 'real', 'sparse'}}


class SNSGAII(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
