# emopylab 2026
"""Fuzzy decision variable framework with various internal optimizers.\n\nReference:\nX. Yang, J. Zou, S. Yang, J. Zheng, and Y. Liu. A fuzzy decision variables framework for large-scale multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2023, 27(3): 445-459. --------------------------------------------------------------------------\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'FDV': {'integer', 'large', 'many', 'multi', 'real'}}


class FDV(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
