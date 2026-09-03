# emopylab 2026
"""MOEA/D with uniform randomly adaptive weights.\n\nReference:\nL. R. C. Farias and A. F. R. Araujo. Many-objective evolutionary algorithm based on decomposition with random and adaptive weights. Proceedings of the IEEE International Conference on Systems, Mans and Cybernetics, 2019.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADURAW': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MOEADURAW(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
