# emopylab 2026
"""Simple indicator-based evolutionary algorithm.\n\nReference:\nE. Zitzler, D. Brockhoff, and L. Thiele. The hypervolume indicator revisited: On the design of Pareto-compliant indicators via weighted integration. Proceedings of the International Conference on Evolutionary Multi-Criterion Optimization, 2007, 862-876.\n"""

from __future__ import annotations

from algorithms.e_moea.e_moea import eMOEA


ALGORITHM_FLAGS = {'SIBEA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class SIBEA(eMOEA):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
