# emopylab 2026
"""MOEA/D with dynamic Thompson sampling.\n\nReference:\nL. Sun and K. Li. Adaptive operator selection based on dynamic Thompson sampling for MOEA/D. Proceedings of the International Conference on Parallel Problem Solving from Nature, 2020, 271-284.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADDYTS': {'integer', 'many', 'multi', 'real'}}


class MOEADDYTS(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
