# emopylab 2026
"""Multi-objective multifactorial evolutionary algorithm.\n\nReference:\nA. Gupta, Y. Ong, L. Feng, and K. C. Tan. Multiobjective multifactorial optimization in evolutionary multitasking. IEEE Transactions on Cybernetics, 2017, 47(7): 1652-1665.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MOMFEA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MOMFEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
