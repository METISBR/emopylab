# emopylab 2026
"""Real-coded genetic algorithm with framework M1-2.\n\nReference:\nK. Deb, R. Hussein, P. C. Roy, and G. Toscano-Pulido. A taxonomy for metamodeling framework for evolutionary multiobjective optimization. IEEE Transactons on Evolutionary Computation, 2019, 23(1): 104-116.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'RGA_M1_2': {'constrained', 'expensive', 'multi', 'real'}}


class RGA_M1_2(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
