# emopylab 2026
"""MOEA/D with fitness-rate-rank-based multiarmed bandit.\n\nReference:\nK. Li, A. Fialho, S. Kwong, and Q. Zhang. Adaptive operator selection with bandits for a multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 2014, 18(1): 114-130.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADFRRMAB': {'integer', 'many', 'multi', 'real'}}


class MOEADFRRMAB(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
