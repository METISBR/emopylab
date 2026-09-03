# emopylab 2026
"""Nested bilevel evolutionary algorithm.\n\nReference:\nA. Sinha, P. Malo, and K. Deb. Test problem construction for single-objective bilevel optimization. Evolutionary Computation, 2014, 22(3): 439-477.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NBLEA': {'bilevel', 'constrained', 'multi', 'real'}}


class NBLEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
