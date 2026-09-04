# emopylab 2026
"""MOEA/D with covariance matrix adaptation evolution strategy.\n\nReference:\nH. Li, Q. Zhang, and J. Deng. Biased multiobjective optimization and decomposition algorithm. IEEE Transactions on Cybernetics, 2017, 47(1): 52-66.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADCMA': {'integer', 'many', 'multi', 'real'}}


class MOEADCMA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
