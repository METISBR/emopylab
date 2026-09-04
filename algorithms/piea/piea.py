# emopylab 2026
"""Performance indicator-based evolutionary algorithm.\n\nReference:\nY. Li, W. Li, S. Li, and Y. Zhao. A performance indicator-based evolutionary algorithm for expensive high-dimensional multi-/many- objective optimization. Information Sciences, 2024: 121045.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'PIEA': {'expensive', 'many', 'multi', 'real'}}


class PIEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
