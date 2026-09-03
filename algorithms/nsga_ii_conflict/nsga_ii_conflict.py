# emopylab 2026
"""NSGA-II with conflict-based partitioning strategy.\n\nReference:\nA. L. Jaimes, C. A. Coello Coello, H. Aguirre, and K. Tanaka. Objective space partitioning using conflict information for solving many-objective problems. Information Sciences, 2014, 268: 305-327.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NSGAIIconflict': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class NSGAIIconflict(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
