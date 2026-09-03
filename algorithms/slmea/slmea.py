# emopylab 2026
"""Super-large-scale multi-objective evolutionary algorithm.\n\nReference:\nY. Tian, Y. Feng, X. Zhang, and C. Sun. A fast clustering based evolutionary algorithm for super-large-scale sparse multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2022, 9(4): 1-16.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SLMEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class SLMEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
