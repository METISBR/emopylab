# emopylab 2026
"""MOEA/D with Pareto front estimation.\n\nReference:\nT. Takagi, K. Takadama, and H. Sato. A multi-objective evolutionary algorithm using weight vector arrangement based on Pareto front estimation. Transaction of the Japanese Society for Evolutionary Computation (Japanese), 2021, 12(2): 45-60.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADPFE': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MOEADPFE(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
