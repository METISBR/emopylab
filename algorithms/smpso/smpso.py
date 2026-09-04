# emopylab 2026
"""Speed-constrained multi-objective particle swarm optimization.\n\nReference:\nA. J. Nebro, J. J. Durillo, J. Garcia-Nieto, C. A. Coello Coello, F. Luna, and E. Alba. SMPSO: A new PSO-based metaheuristic for multi-objective optimization. Proceedings of the IEEE Symposium on Computational Intelligence in Multi-Criteria Decision-Making, 2009, 66-73.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SMPSO': {'integer', 'multi', 'real'}}


class SMPSO(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
