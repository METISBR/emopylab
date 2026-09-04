# emopylab 2026
"""Memetic algorithm with Pareto archived evolution strategy.\n\nReference:\nJ. D. Knowles and D. W. Corne. M-PAES: A memetic algorithm for multiobjective optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2000, 325-332.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MPAES': {'integer', 'multi', 'real'}}


class MPAES(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
