# emopylab 2026
"""Dynamic selection preference-assisted constrained multiobjective differential evolution.\n\nReference:\nK. Yu, J. Liang, B. Qu, Y. Luo, and C. Yue. Dynamic selection preference-assisted constrained multiobjective differential evolution. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(5): 2954-2965.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'DSPCMDE': {'constrained', 'integer', 'multi', 'real'}}


class DSPCMDE(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
