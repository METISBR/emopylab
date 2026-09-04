# emopylab 2026
"""Multi-objective neural architecture search.\n\nReference:\nF. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin. An evolutionary framework for multi-objective neural architecture search. IEEE Transactions on Evolutionary Computation, 2025.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MONAS': {'binary', 'integer', 'label', 'multi', 'multimodal', 'permutation', 'real'}}


class MONAS(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
