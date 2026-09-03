# emopylab 2026
"""Subspace segmentation based co-evolutionary algorithm.\n\nReference:\nG. Liu, Z. Pei, N. Liu, and Y. Tian. Subspace segmentation based co-evolutionary algorithm for balancing convergence and diversity in many-objective optimization. Swarm and Evolutionary Computation, 2023, 83: 101410.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SSCEA': {'integer', 'many', 'multi', 'real'}}


class SSCEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
