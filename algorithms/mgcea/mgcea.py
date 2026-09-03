# emopylab 2026
"""Multi-granularity clustering based evolutionary algorithm.\n\nReference:\nY. Tian, S. Shao, G. Xie, and Y. Jin. A multi-granularity clustering based evolutionary algorithm for large-scale sparse multi-objective optimization. Swarm and Evolutionary Computation, 2024, 84: 101453.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MGCEA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class MGCEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
