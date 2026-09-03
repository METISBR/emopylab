# emopylab 2026
"""AGE-II (Approximation-guided evolutionary algorithm II).

Reference:
M. Wagner and F. Neumann. A fast approximation-guided evolutionary multi-objective algorithm. Proceedings of GECCO, 2013, 687-694.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'AGEII': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class AGEII(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, epsilon=0.1, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.epsilon = epsilon
