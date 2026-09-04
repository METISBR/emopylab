# emopylab 2026
"""Efficient global optimization for Pareto optimization.\n\nReference:\nJ. Knowles. ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2006, 10(1): 50-66.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'ParEGO': {'expensive', 'integer', 'multi', 'real'}}


class ParEGO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
