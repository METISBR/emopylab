# emopylab 2026
"""Efficient dropout neural network based AR-MOEA.\n\nReference:\nD. Guo, X. Wang, K. Gao, Y. Jin, J. Ding, and T. Chai. Evolutionary optimization of high-dimensional multiobjective and many-objective expensive problems assisted by a dropout neural network. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(4): 2084-2097.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'EDNARMOEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


class EDNARMOEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
