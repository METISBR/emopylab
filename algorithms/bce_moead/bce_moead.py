# emopylab 2026
"""BCE-MOEA-D (Bi-criterion evolution based MOEA/D).

Reference:
M. Li, S. Yang, and X. Liu. Pareto or non-Pareto: Bi-criterion evolution in multiobjective optimization. IEEE TEC, 2016, 20(5): 645-665.
"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'BCEMOEAD': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class BCEMOEAD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, bicriterion=True, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.bicriterion = bicriterion
