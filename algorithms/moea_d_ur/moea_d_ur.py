# emopylab 2026
"""MOEA/D with update when required.\n\nReference:\nL. R. de Farias and A. F. Araujo. A decomposition-based many-objective evolutionary algorithm updating weights when required. Swarm and Evolutionary Computation, 2022, 68: 100980.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADUR': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MOEADUR(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
