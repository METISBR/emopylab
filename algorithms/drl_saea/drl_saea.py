# emopylab 2026
"""Deep reinforcement learning-based expensive constrained evolutionary algorithm.\n\nReference:\nS. Shao, Y. Tian, and Y. Zhang. Deep reinforcement learning assisted surrogate model management for expensive constrained multi-objective optimization. Swarm and Evolutionary Computation, 2025, 92: 101817.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DRLSAEA': {'constrained', 'expensive', 'multi', 'real'}}


class DRLSAEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
