# emopylab 2026
"""Decomposition-based multi-objective evolutionary algorithm with the.\n\nReference:\nJ. Chen, J. Li, and B. Xin. DMOEA-eC: Decomposition-based multiobjective evolutionary algorithm with the e-constraint framework. IEEE Transactions on Evolutionary Computation, 2017, 21(5): 714-730.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'DMOEAeC': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class DMOEAeC(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
