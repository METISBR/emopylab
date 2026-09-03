# emopylab 2026
"""SPEA2 with shift-based density estimation.\n\nReference:\nM. Li, S. Yang, and X. Liu. Shift-based density estimation for Pareto-based algorithms in many-objective optimization. IEEE Transactions on Evolutionary Computation, 2014, 18(3): 348-365.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SPEA2SDE': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class SPEA2SDE(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
