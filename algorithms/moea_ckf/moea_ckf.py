# emopylab 2026
"""Multi-objective evolutionary algorithm based on cross-scale knowledge fusion.\n\nReference:\nZ. Ding, L. Chen, D. Sun, and X. Zhang. Efficient sparse large-scale multi-objective optimization based on cross-scale knowledge fusion. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(11): 6989-7001.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MOEACKF': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class MOEACKF(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
