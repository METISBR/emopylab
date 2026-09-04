# emopylab 2026
"""S-metric-selection-based efficient global optimization.\n\nReference:\nW. Ponweiser, T. Wagner, D. Biermann, and M. Vincze. Multiobjective optimization on a limited budget of evaluations using model-assisted S-metric selection. Proceedings of the International Conference on Parallel Problem Solving from Nature, 2008, 784-794.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'SMSEGO': {'expensive', 'integer', 'multi', 'real'}}


class SMSEGO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
