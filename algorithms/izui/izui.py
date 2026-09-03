# emopylab 2026
"""An aggregative gradient based multi-objective optimizer proposed by Izui et al..\n\nReference:\nK. Izui, T. Yamada, S. Nishiwaki, and K. Tanaka. Multiobjective optimization using an aggregative gradient-based method. Structural and Multidisciplinary Optimization, 2015, 51: 173-182.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'Izui': {'constrained', 'large', 'many', 'multi', 'real'}}


class Izui(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
