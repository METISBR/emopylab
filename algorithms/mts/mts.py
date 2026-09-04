# emopylab 2026
"""Multiple trajectory search.\n\nReference:\nL. Y. Tseng and C. Chen. Multiple trajectory search for unconstrained / constrained multi-objective optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2009, 1951-1958.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MTS': {'integer', 'multi', 'real'}}


class MTS(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
