# emopylab 2026
"""Probability and mapping crowding distance.\n\nReference:\nY. Li, W. Li, Y. Zhao, and S. Li. An infill sampling criterion based on improvement of probability and mapping crowding distance for expensive multi/many-objective optimization. Engineering Applications of Artificial Intelligence, 2024, 133: 108616.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'PIMD': {'expensive', 'many', 'multi', 'real'}}


class PIMD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
