# emopylab 2026
"""Multi-stage multi-objective evolutionary algorithm.\n\nReference:\nY. Tian, C. He, R. Cheng, and X. Zhang. A multi-stage evolutionary algorithm for better diversity preservation in multi-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2021, 51(9): 5880-5894.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MSEA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MSEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
