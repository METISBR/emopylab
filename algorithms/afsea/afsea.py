# emopylab 2026
"""AFSEA (Adjoint feature-selection evolutionary algorithm).

Reference:
P. Zhang, H. Yin, Y. Tian, and X. Zhang. An adjoint feature-selection-based evolutionary algorithm for sparse large-scale multiobjective optimization. Complex & Intelligent Systems, 2025, 11:127.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'AFSEA': {'binary', 'integer', 'multi', 'real', 'sparse'}}


class AFSEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
