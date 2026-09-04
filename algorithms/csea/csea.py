# emopylab 2026
"""Classification based surrogate-assisted evolutionary algorithm.\n\nReference:\nL. Pan, C. He, Y. Tian, H. Wang, X. Zhang, and Y. Jin. A classification based surrogate-assisted evolutionary algorithm for expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2019, 23(1): 74-88.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'CSEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


class CSEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
