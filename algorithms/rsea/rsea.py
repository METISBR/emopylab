# emopylab 2026
"""Radial space division based evolutionary algorithm.\n\nReference:\nC. He, Y. Tian, Y. Jin, X. Zhang, and L. Pan. A radial space division based evolutionary algorithm for many-objective optimization. Applied Soft Computing, 2017, 61: 603-621.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'RSEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class RSEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
