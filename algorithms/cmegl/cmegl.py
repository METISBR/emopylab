# emopylab 2026
"""Constrained evolutionary multitasking with global and local auxiliary tasks.

Reference:
K. Qiao, J. Liang, Z. Liu, K. Yu, C. Yue, and B. Qu. Evolutionary multitasking with global and local auxiliary tasks for constrained multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2023, 10(10): 1951-1964.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'CMEGL': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CMEGL(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
