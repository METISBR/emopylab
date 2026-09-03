# emopylab 2026
"""Problem reformulation and duplication handling.\n\nReference:\nR. Jiao, B. Xue, and M. Zhang. Solving multiobjective feature selection problems in classification via problem reformulation and duplication handling. IEEE Transactions on Evolutionary Computation, 2024, 28(4): 846-860.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'PRDH': {'binary', 'multi'}}


class PRDH(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
