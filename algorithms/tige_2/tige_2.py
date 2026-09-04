# emopylab 2026
"""Tri-Goal Evolution Framework for CMaOPs.\n\nReference:\nY. Zhou, Z. Min, J. Wang, Z. Zhang, and J.Zhang. Tri-goal evolution framework for constrained many-objective optimization. IEEE Transactions on Systems Man and Cybernetics Systems, 2020, 50(8): 3086-3099.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'TiGE2': {'binary', 'constrained', 'integer', 'label', 'many', 'permutation', 'real'}}


class TiGE2(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
