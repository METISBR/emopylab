# emopylab 2026
"""Kriging-assisted Two_Arch2.\n\nReference:\nZ. Song, H. Wang, C. He, and Y. Jin. A Kriging-assisted two-archive evolutionary algorithm for expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(6): 1013-1027.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'KTA2': {'expensive', 'integer', 'many', 'multi', 'real'}}


class KTA2(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
