# emopylab 2026
"""Reference points-based evolutionary algorithm.\n\nReference:\nalpha --- 0.4 --- Ratio of individuals being used to generate reference points delta --- 0.1 --- Parameter determining the difference between the reference point and the individuals Y. Liu, D. Gong, X. Sun, and Y. Zhang. Many-objective evolutionary optimization based on reference points. Applied Soft Computing, 2017, 50: 344-355.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'RPEA': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class RPEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
