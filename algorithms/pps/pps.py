# emopylab 2026
"""Push and pull search algorithm.\n\nReference:\nZ. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, and E. Goodman. Push and pull search for solving constrained multi-objective optimization problems. Swarm and Evolutionary Computation, 2019, 44(2): 665-679.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'PPS': {'constrained', 'integer', 'many', 'multi', 'real'}}


class PPS(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
