# emopylab 2026
"""Hyper-dominance based evolutionary algorithm.\n\nReference:\nZ. Liu, F. Han, Q. Ling, H. Han, and J. Jiang. A many-objective optimization evolutionary algorithm based on hyper-dominance degree. Swarm and Evolutionary Computation, 2023, 83: 101411.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'HEA': {'binary', 'many', 'multi', 'permutation', 'real'}}


class HEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
