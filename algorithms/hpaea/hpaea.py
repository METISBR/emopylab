# emopylab 2026
"""Hyperplane assisted evolutionary algorithm.\n\nReference:\nH. Chen, Y. Tian, W. Pedrycz, G. Wu, R. Wang, and L. Wang. Hyperplane assisted evolutionary algorithm for many-objective optimization problems. IEEE Transactions on Cybernetics, 2020, 50(7): 3367-3380.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'hpaEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class hpaEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
