# emopylab 2026
"""Multiobjective PSO using ring topology and special crowding distance.\n\nReference:\nC. Yue, B. Qu, and J. Liang. A multiobjective particle swarm optimizer using ring topology for solving multimodal multiobjective problems. IEEE Transactions on Evolutionary Computation, 2018, 22(5): 805-817.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MO_Ring_PSO_SCD': {'integer', 'multi', 'multimodal', 'real'}}


class MO_Ring_PSO_SCD(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
