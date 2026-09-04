# emopylab 2026
"""Expensive multiobjective optimization by relation learning and prediction.\n\nReference:\nH. Hao, A. Zhou, H. Qian, and H. Zhang. Expensive multiobjective optimization by relation learning and prediction. IEEE Transactions on Evolutionary Computation, 2022, 26(5): 1157-1170.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'REMO': {'expensive', 'many', 'multi', 'real'}}


class REMO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
