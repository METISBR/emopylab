# emopylab 2026
"""Many-objective evolutionary algorithm based on objective space reduction.\n\nReference:\nZ. He and G. G. Yen. Many-objective evolutionary algorithm: Objective space reduction and diversity improvement. IEEE Transactions on Evolutionary Computation, 2016, 20(1): 145-160.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MaOEARD': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class MaOEARD(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
