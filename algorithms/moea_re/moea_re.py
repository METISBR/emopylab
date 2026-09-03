# emopylab 2026
"""Multi-objective evolutionary algorithm with robustness enhancement.\n\nReference:\nZ. He, G. G. Yen, and J. Lv. Evolutionary multiobjective optimization with robustness enhancement. IEEE Transactions on Evolutionary Computation, 2020, 24(3): 494-507.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEARE': {'binary', 'integer', 'label', 'multi', 'permutation', 'real', 'robust'}}


class MOEARE(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
