# emopylab 2026
"""Expected improvement matrix based efficient global optimization.\n\nReference:\nD. Zhan, Y. Cheng, and J. Liu. Expected improvement matrix-based infill criteria for expensive multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2017, 21(6): 956-975.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'EIMEGO': {'expensive', 'integer', 'multi', 'real'}}


class EIMEGO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
