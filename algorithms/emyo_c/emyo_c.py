# emopylab 2026
"""Evolutionary many-objective optimization algorithm with clustering-based.\n\nReference:\nR. Denysiuk, L. Costa, and I. E. Santo. Clustering-based selection for evolutionary many-objective optimization. Proceedings of the International Conference on Parallel Problem Solving from Nature, 2014, 538-547.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'EMyOC': {'integer', 'many', 'multi', 'real'}}


class EMyOC(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
