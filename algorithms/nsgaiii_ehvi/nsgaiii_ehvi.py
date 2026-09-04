# emopylab 2026
"""NSGA-III with expected hypervolume improvement.\n\nReference:\nY. Pang, Y. Wang, S. Zhang, X. Lai, W. Sun, and X. Song. An expensive many-objective optimization algorithm based on efficient expected hypervolume improvement. IEEE Transactions on Evolutionary Computation, 2023, 27(6): 1822-1836.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'NSGAIIIEHVI': {'expensive', 'many', 'multi', 'real'}}


class NSGAIIIEHVI(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
