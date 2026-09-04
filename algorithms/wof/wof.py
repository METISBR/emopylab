# emopylab 2026
"""Weighted optimization framework.\n\nReference:\nH. Zille, H. Ishibuchi, S. Mostaghim, and Y. Nojima. A framework for large-scale multiobjective optimization based on problem transformation. IEEE Transactions on Evolutionary Computation, 2018, 22(2): 260-275. -----------------------------------------------------------------------\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'WOF': {'integer', 'large', 'multi', 'real'}}


class WOF(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
