# emopylab 2026
"""Knowledge learning based NSGA-II.\n\nReference:\nQ. Zhao, B. Yan, Y. Shi, and M. Middendorf. Evolutionary dynamic multiobjective optimization via learning from historical search process. IEEE Transactions on Cybernetics, 2021, 52(7): 6119-6130.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'KLNSGAII': {'binary', 'constrained', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


class KLNSGAII(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
