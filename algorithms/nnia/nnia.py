# emopylab 2026
"""Nondominated neighbor immune algorithm.\n\nReference:\nM. Gong, L. Jiao, H. Du, and L. Bo. Multiobjective immune algorithm with nondominated neighbor-based selection. Evolutionary Computation, 2008, 16(2): 225-255.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NNIA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class NNIA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
