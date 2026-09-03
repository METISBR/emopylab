# emopylab 2026
"""MOEA/D with adaptive weight adjustment.\n\nReference:\nY. Qi, X. Ma, F. Liu, L. Jiao, J. Sun, and J. Wu. MOEA/D with adaptive weight adjustment. Evolutionary Computation, 2014, 22(2): 231-264.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADAWA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MOEADAWA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
