# emopylab 2026
"""Large-scale multi-objective optimization framework with NSGA-II.\n\nReference:\nC. He, L. Li, Y. Tian, X. Zhang, R. Cheng, Y. Jin, and X. Yao. Accelerating large-scale multi-objective optimization via problem reformulation. IEEE Transactions on Evolutionary Computation, 2019, 23(6): 949-961.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'LSMOF': {'integer', 'large', 'multi', 'real'}}


class LSMOF(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
