# emopylab 2026
"""MOEA/D with efficient global optimization.\n\nReference:\nQ. Zhang, W. Liu, E. Tsang, and B. Virginas. Expensive multiobjective optimization by MOEA/D with Gaussian process model. IEEE Transactions on Evolutionary Computation, 2010, 14(3): 456-474.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADEGO': {'expensive', 'integer', 'multi', 'real'}}


class MOEADEGO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
