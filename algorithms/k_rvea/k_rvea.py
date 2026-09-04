# emopylab 2026
"""Surrogate-assisted RVEA.\n\nReference:\nT. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya. A surrogate- assisted reference vector guided evolutionary algorithm for computationally expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2018, 22(1): 129-142.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'KRVEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


class KRVEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
