# emopylab 2026
"""Indicator-based evolutionary algorithm.\n\nReference:\nE. Zitzler and S. Kunzli. Indicator-based selection in multiobjective search. Proceedings of the International Conference on Parallel Problem Solving from Nature, 2004, 832-842.\n"""

from __future__ import annotations

from algorithms.e_moea.e_moea import eMOEA


ALGORITHM_FLAGS = {'IBEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class IBEA(eMOEA):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
