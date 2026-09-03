# emopylab 2026
"""Hybrid hierarchical clustering based multi-modal multi-objective evolutionary algorithm.\n\nReference:\nZ. Ding, L. Cao, L. Chen, D. Sun, X. Zhang, and Z. Tao. Large-scale multimodal multiobjective evolutionary optimization based on hybrid hierarchical clustering. Knowledge-Based Systems, 2023, 266: 110398.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'HHCMMEA': {'large', 'multi', 'multimodal', 'real', 'sparse'}}


class HHCMMEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
