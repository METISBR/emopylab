# emopylab 2026
"""Tri-population based push and pull search.\n\nReference:\nF. Ming, W. Gong, L. Wang, and C. Lu. A tri-population based co-evolutionary framework for constrained multi-objective optimization problems. Swarm and Evolutionary Computation, 2022, 70: 101055.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DPPPS': {'constrained', 'multi', 'real'}}


class DPPPS(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
