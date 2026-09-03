# emopylab 2026
"""Knowledge learning-based evolutionary algorithm.\n\nReference:\nS. Shao, Y. Tian, Y. Zhang, and X. Zhang. Knowledge learning-based dimensionality reduction for solving large-scale sparse multiobjective optimization problems. IEEE Transactions on Cybernetics, 2025, 55(7): 3471-3484.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'KLEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class KLEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
