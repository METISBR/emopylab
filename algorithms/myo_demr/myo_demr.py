# emopylab 2026
"""Many-objective differential evolution with mutation restriction.\n\nReference:\nR. Denysiuk, L. Costa, and I. E. Santo. Many-objective optimization using differential evolution with variable-wise mutation restriction. Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2013, 591-598.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MyODEMR': {'integer', 'many', 'multi', 'real'}}


class MyODEMR(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
