# emopylab 2026
"""MOPSO based on decomposition.\n\nReference:\nS. Z. Martinez and C. A. Coello Coello. A multi-objective particle swarm optimizer based on decomposition. Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2011, 69-76.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'dMOPSO': {'integer', 'multi', 'real'}}


class dMOPSO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
