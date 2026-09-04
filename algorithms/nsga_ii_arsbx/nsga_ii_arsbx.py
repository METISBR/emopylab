# emopylab 2026
"""NSGA-II with adaptive rotation based simulated binary crossover.\n\nReference:\nL. Pan, W. Xu, L. Li, C. He, and R. Cheng. Adaptive simulated binary crossover for rotated multi-objective optimization. Swarm and Evolutionary Computation, 2021, 60: 100759. --------------------------------------------------------------------------\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'NSGAIIARSBX': {'constrained', 'integer', 'multi', 'real'}}


class NSGAIIARSBX(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
