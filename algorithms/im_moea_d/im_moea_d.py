# emopylab 2026
"""Inverse modeling MOEA/D.\n\nReference:\nL. R. C. Farias and A. F. R. Araujo. IM-MOEA/D: An inverse modeling multi-objective evolutionary algorithm based on decomposition. Proceedings of the IEEE International Conference on Systems, Mans and Cybernetics, 2021.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'IMMOEAD': {'integer', 'large', 'multi', 'real'}}


class IMMOEAD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
