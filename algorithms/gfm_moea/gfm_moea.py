# emopylab 2026
"""Generic front modeling based multi-objective evolutionary algorithm.\n\nReference:\nY. Tian, X. Zhang, R. Cheng, C. He, and Y. Jin. Guiding evolutionary multi-objective optimization with generic front modeling. IEEE Transactions on Cybernetics, 2020, 50(3): 1106-1119.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'GFMMOEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class GFMMOEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
