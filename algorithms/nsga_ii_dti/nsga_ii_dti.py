# emopylab 2026
"""NSGA-II of Deb's type I robust version.\n\nReference:\nK. Deb and H. Gupta. Introducing robustness in multi-objective optimization. Evolutionary Computation, 2006, 14(4): 463-494.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NSGAIIDTI': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real', 'robust'}}


class NSGAIIDTI(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
