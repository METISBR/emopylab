# emopylab 2026
"""Multiobjective evolutionary algorithm based on polar coordinates.\n\nReference:\nR. Denysiuk, L. Costa, I. E. Santo, and J. C. Matos. MOEA/PC: Multiobjective evolutionary algorithm based on polar coordinates. Proceedings of the International Conference on Evolutionary Multi- Criterion Optimization, 2015, 141-155.\n"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'MOEAPC': {'integer', 'multi', 'real'}}


class MOEAPC(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
