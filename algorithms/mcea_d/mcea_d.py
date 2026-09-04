# emopylab 2026
"""Multiple classifiers-assisted evolutionary algorithm based on decomposition.\n\nReference:\nT. Sonoda and M. Nakata. Multiple classifiers-assisted evolutionary algorithm based on decomposition for high-dimensional multi-objective problems. IEEE Transactions on Evolutionary Computation, 2022, 26(6): 1581-1595.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MCEAD': {'expensive', 'integer', 'many', 'multi', 'real'}}


class MCEAD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
