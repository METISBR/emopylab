# emopylab 2026
"""Non-uniform clustering based evolutionary algorithm.\n\nReference:\nS. Shao, Y. Tian, and X. Zhang. A non-uniform clustering based evolutionary algorithm for solving large-scale sparse multi-objective optimization problems. Proceedings of the 18th International Conference on Bio-inspired Computing: Theories and Applications, 2023.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NUCEA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class NUCEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
