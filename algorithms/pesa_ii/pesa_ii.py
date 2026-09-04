# emopylab 2026
"""Pareto envelope-based selection algorithm II.\n\nReference:\nD. W. Corne, N. R. Jerram, J. D. Knowles, and M. J. Oates. PESA-II: Region-based selection in evolutionary multiobjective optimization. Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2001, 283-290.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'PESAII': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class PESAII(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
