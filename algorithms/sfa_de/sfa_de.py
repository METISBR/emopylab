# emopylab 2026
"""Scalarization function approximation based differential evolution algorithm.\n\nReference:\nY. Horaguchi, K. Nishihara, and M. Nakata. Evolutionary multiobjective optimization assisted by scalarization function approximation for high-dimensional expensive problems. Swarm and Evolutionary Computation, 2024, 86: 101516.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'SFADE': {'expensive', 'integer', 'many', 'multi', 'real'}}


class SFADE(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
