# emopylab 2026
"""Multi-objective particle swarm optimization.\n\nReference:\nC. A. Coello Coello and M. S. Lechuga. MOPSO: A proposal for multiple objective particle swarm optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2002, 1051-1056.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOPSO': {'integer', 'multi', 'real'}}


class MOPSO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
