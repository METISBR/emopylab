# emopylab 2026
"""MOEA/D with stable matching.\n\nReference:\nK. Li, Q. Zhang, S. Kwong, M. Li, and R. Wang. Stable matching-based selection in evolutionary multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2014, 18(6): 909-923.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADSTM': {'integer', 'many', 'multi', 'real'}}


class MOEADSTM(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
