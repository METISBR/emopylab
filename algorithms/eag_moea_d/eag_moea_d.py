# emopylab 2026
"""External archive guided MOEA/D.\n\nReference:\nX. Cai, Y. Li, Z. Fan, and Q. Zhang. An external archive guided multiobjective evolutionary algorithm based on decomposition for combinatorial optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 508-523.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'EAGMOEAD': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class EAGMOEAD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
