# emopylab 2026
"""Global weighting achievement scalarizing function genetic algorithm.\n\nReference:\nR. Saborido, A. B. Ruiz, and M. Luque. Global WASF-GA: An evolutionary algorithm in multiobjective optimization to approximate the whole Pareto optimal front. Evolutionary computation, 2017, 25(2): 309-349.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'GWASFGA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class GWASFGA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
