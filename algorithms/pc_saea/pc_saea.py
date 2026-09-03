# emopylab 2026
"""Pairwise comparison based surrogate-assisted evolutionary algorithm.\n\nReference:\nY. Tian, J. Hu, C. He, H. Ma, L. Zhang, and X. Zhang. A pairwise comparison based surrogate-assisted evolutionary algorithm for expensive multi-objective optimization. Swarm and Evolutionary Computation, 2023, 80: 101323.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'PCSAEA': {'expensive', 'many', 'multi', 'real'}}


class PCSAEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
