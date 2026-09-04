# emopylab 2026
"""Multi-objective evolutionary gradient search.\n\nReference:\nC. K. Goh, Y. S. Ong, K. C. Tan, and E. J. Teoh. An investigation on evolutionary gradient search for multi-objective optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2008.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MOEGS': {'large', 'multi', 'real'}}


class MOEGS(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
