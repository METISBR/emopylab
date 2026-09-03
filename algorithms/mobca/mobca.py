# emopylab 2026
"""Multi-objective besiege and conquer algorithm.\n\nReference:\nJ. Jiang, J. Wu, J. Luo, X. Yang, and Z. Huang. MOBCA: multi-objective besiege and conquer algorithm. Biomimetics, 2024, 9: 316.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOBCA': {'integer', 'multi', 'real'}}


class MOBCA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
