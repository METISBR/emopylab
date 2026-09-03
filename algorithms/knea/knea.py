# emopylab 2026
"""Knee point driven evolutionary algorithm.\n\nReference:\nX. Zhang, Y. Tian, and Y. Jin. A knee point-driven evolutionary algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(6): 761-776.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'KnEA': {'binary', 'constrained', 'integer', 'label', 'many', 'permutation', 'real'}}


class KnEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
