# emopylab 2026
"""MOEA/D with maximum relative diversity loss.\n\nReference:\nS. B. Gee, K. C. Tan, V. A. Shim, and N. R. Pal. Online diversity assessment in evolutionary multiobjective optimization: A geometrical perspective. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 542-559.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADMRDL': {'integer', 'multi', 'real'}}


class MOEADMRDL(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
