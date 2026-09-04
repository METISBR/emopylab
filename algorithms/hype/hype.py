# emopylab 2026
"""Hypervolume estimation algorithm.\n\nReference:\nJ. Bader and E. Zitzler. HypE: An algorithm for fast hypervolume-based many-objective optimization. Evolutionary Computation, 2011, 19(1): 45-76.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'HypE': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class HypE(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
