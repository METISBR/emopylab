# emopylab 2026
"""Many-objective evolutionary algorithm based on directional diversity and.\n\nReference:\nJ. Cheng, G. G. Yen, and G. Zhang. A many-objective evolutionary algorithm with enhanced mating and environmental selections. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 592-605.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MaOEADDFC': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MaOEADDFC(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
