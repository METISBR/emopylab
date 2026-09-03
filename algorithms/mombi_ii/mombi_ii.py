# emopylab 2026
"""Many objective metaheuristic based on the R2 indicator II.\n\nReference:\nR. Hernandez Gomez and C. A. Coello Coello. Improved metaheuristic based on the R2 indicator for many-objective optimization. Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2015, 679-686.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOMBIII': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MOMBIII(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
