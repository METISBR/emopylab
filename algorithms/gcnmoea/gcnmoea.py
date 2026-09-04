# emopylab 2026
"""Graph convolutional network based multi-objective evolutionary algorithm.\n\nReference:\nP. Yan, Y. Tian, and Y. Liu. An indicator-based multi-objective evolutionary algorithm assisted by improved graph convolutional networks. Swarm and Evolutionary Computation, 2025, 94: 101892.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'GCNMOEA': {'integer', 'multi', 'real'}}


class GCNMOEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
