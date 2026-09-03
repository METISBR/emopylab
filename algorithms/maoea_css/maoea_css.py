# emopylab 2026
"""Many-objective evolutionary algorithms based on coordinated selection.\n\nReference:\nZ. He and G. G. Yen. Many-objective evolutionary algorithms based on coordinated selection strategy. IEEE Transactions on Evolutionary Computation, 2017, 21(2): 220-233.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MaOEACSS': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MaOEACSS(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
