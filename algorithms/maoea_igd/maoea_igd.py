# emopylab 2026
"""IGD based many-objective evolutionary algorithm.\n\nReference:\nY. Sun, G. G. Yen, and Z. Yi. IGD indicator-based evolutionary algorithm for many-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2019, 23(2): 173-187.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MaOEAIGD': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class MaOEAIGD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
