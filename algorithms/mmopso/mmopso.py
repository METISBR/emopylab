# emopylab 2026
"""MOPSO with multiple search strategies.\n\nReference:\nQ. Lin, J. Li, Z. Du, J. Chen, and Z. Ming. A novel multi-objective particle swarm optimization with multiple search strategies. European Journal of Operational Research, 2015, 247(3): 732-744.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MMOPSO': {'integer', 'multi', 'real'}}


class MMOPSO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
