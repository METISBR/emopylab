# emopylab 2026
"""NSGA-III based on Pareto based bi-indicator infill sampling criterion.\n\nReference:\nZ. Song, H. Wang, and H. Xu. A framework for expensive many-objective optimization with Pareto-based bi-indicator infill sampling criterion. Memetic Computing, 2022, 14: 179-191.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'PBNSGAIII': {'expensive', 'integer', 'many', 'multi', 'real'}}


class PBNSGAIII(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
