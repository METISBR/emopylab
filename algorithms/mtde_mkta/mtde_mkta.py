# emopylab 2026
"""Multitasking differential evolution with multiple knowledge types and transfer adaptation.\n\nReference:\nY. Li and W. Gong. Multiobjective multitask optimization with multiple knowledge types and transfer adaptation. IEEE Transactions on Evolutionary Computation, 2025, 29(1): 205-216.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MTDEMKTA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MTDEMKTA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
