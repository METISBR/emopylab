# emopylab 2026
"""Bilevel evolutionary algorithm based on quadratic approximations II.\n\nReference:\nA. Sinha, Z. Lu, K. Deb, and P. Malo. Bilevel optimization based on iterative approximation of mappings. Journal of Heuristics, 2020, 26: 151-185.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'BLEAQII': {'bilevel', 'constrained', 'multi', 'real'}}


class BLEAQII(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
