# emopylab 2026
"""Expensive multi-/many-objective evolutionary algorithm.\n\nReference:\nS. Qin, C. Sun, Q. Liu, and Y. Jin. A performance indicator-based infill criterion for expensive multi-/many-objective optimization. IEEE Transactions on Evolutionary Computation, 2023, 27(4): 1085-1099.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'EMMOEA': {'expensive', 'integer', 'multi', 'real'}}


class EMMOEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
