# emopylab 2026
"""CA-MOEA (Clustering-based adaptive MOEA).

Reference:
Y. Hua, Y. Jin, and K. Hao. A clustering-based adaptive evolutionary algorithm for multiobjective optimization with irregular Pareto fronts. IEEE Transactions on Cybernetics, 2019, 49(7): 2758-2770.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'CAMOEA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CAMOEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
