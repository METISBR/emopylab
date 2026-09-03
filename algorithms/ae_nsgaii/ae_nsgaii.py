# emopylab 2026
"""AE-NSGA-II (Autoencoding NSGA-II).

Reference:
L. Feng, W. Zhou, W. Liu, Y. S. Ong, and K. C. Tan. Solving dynamic multiobjective problem via autoencoding evolutionary search. IEEE Transactions on Cybernetics, 2020, 52(5): 2649-2662.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'AENSGAII': {'binary', 'constrained', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


class AENSGAII(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
