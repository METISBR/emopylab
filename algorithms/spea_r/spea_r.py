# emopylab 2026
"""Strength Pareto evolutionary algorithm based on reference direction.\n\nReference:\nS. Jiang and S. Yang. A strength Pareto evolutionary algorithm based on reference direction for multiobjective and many-objective optimization. IEEE Transactions on Evolutionary Computation, 2017, 21(3): 329-346.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'SPEAR': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class SPEAR(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
