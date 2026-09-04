# emopylab 2026
"""Multi-objective particle swarm optimization algorithm based on.\n\nReference:\nC. Dai, Y. Wang, and M. Ye. A new multi-objective particle swarm optimization algorithm based on decomposition. Information Sciences, 2015, 325: 541-557.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MPSOD': {'integer', 'many', 'multi', 'real'}}


class MPSOD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
