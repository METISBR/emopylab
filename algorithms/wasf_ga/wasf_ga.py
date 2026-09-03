# emopylab 2026
"""Weighting achievement scalarizing function genetic algorithm.\n\nReference:\nA. B. Ruiz, R. Saborido, and M. Luque. A preference-based evolutionary algorithm for multiobjective optimization: the weighting achievement scalarizing function genetic algorithm. Journal of Global Optimization, 2015, 62: 101-129.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'WASFGA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class WASFGA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
