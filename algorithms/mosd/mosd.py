# emopylab 2026
"""Multiobjective steepest descent.\n\nReference:\nX. Liu and A. C. Reynolds. A multiobjective steepest descent method with applications to optimal well control. Computational Geosciences, 2016, 20: 355-374.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MOSD': {'constrained', 'large', 'multi', 'real'}}


class MOSD(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
