# emopylab 2026
"""Supervised multi-objective optimization algorithm.\n\nReference:\nT. Takagi, K. Takadama, and H. Sato. Supervised multi-objective optimization algorithm using estimation. Proceedings of the IEEE Congress on Evolutionary Computation, 2022.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'SMOA': {'expensive', 'multi', 'real'}}


class SMOA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
