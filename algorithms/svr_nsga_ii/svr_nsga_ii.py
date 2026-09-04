# emopylab 2026
"""Support vector regression based NSGA-II.\n\nReference:\nL. Cao, L. Xu, E. D. Goodman, C. Bao, and S. Zhu. Evolutionary dynamic multiobjective optimization assisted by a support vector regression predictor. IEEE Transactions on Evolutionary Computation, 2019, 24(2): 305-319.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SVRNSGAII': {'binary', 'constrained', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


class SVRNSGAII(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
