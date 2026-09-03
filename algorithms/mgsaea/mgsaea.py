# emopylab 2026
"""Multigranularity surrogate-assisted constrained evolutionary algorithm.\n\nReference:\nY. Zhang, H. Jiang, Y. Tian, H. Ma, and X. Zhang. Multigranularity surrogate modeling for evolutionary multiobjective optimization with expensive constraints. IEEE Transactions on Neural Networks and Learning Systems, 2024, 35(3): 2956-2968.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MGSAEA': {'constrained', 'expensive', 'multi', 'real'}}


class MGSAEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
