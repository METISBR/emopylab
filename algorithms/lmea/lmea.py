# emopylab 2026
"""Evolutionary algorithm for large-scale many-objective optimization.\n\nReference:\nX. Zhang, Y. Tian, R. Cheng, and Y. Jin. A decision variable clustering based evolutionary algorithm for large-scale many-objective optimization. IEEE Transactions on Evolutionary Computation, 2018, 22(1): 97-112.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'LMEA': {'integer', 'large', 'many', 'multi', 'real'}}


class LMEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
