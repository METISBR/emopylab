# emopylab 2026
"""Two-archive algorithm 2.\n\nReference:\nH. Wang, L. Jiao, and X. Yao. Two_Arch2: An improved two-archive algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 524-541.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'Two_Arch2': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class Two_Arch2(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
