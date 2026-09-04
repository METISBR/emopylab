# emopylab 2026
"""Multi-objective linear subspace surrogate modeling assisted evolutionary algorithm.\n\nReference:\nL. Si, X. Zhang, Y. Tian, S. Yang, L. Zhang, and Y. Jin. Linear subspace surrogate modeling for large-scale expensive single/multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2025, 29(3): 697-710.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOL2SMEA': {'expensive', 'large', 'multi', 'real'}}


class MOL2SMEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
