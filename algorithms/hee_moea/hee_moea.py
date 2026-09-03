# emopylab 2026
"""Multiobjective evolutionary algorithm with heterogeneous ensemble based.\n\nReference:\nD. Guo, Y. Jin, J. Ding, and T. Chai. Heterogeneous ensemble-based infill criterion for evolutionary multiobjective optimization of expensive problems. IEEE Transactions on Cybernetics, 2019, 49(3): 1012-1025.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'HeEMOEA': {'expensive', 'integer', 'multi', 'real'}}


class HeEMOEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
