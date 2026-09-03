# emopylab 2026
"""Multi-modal MOEA using two-archive and recombination strategies.\n\nReference:\nY. Liu, G. G. Yen, and D. Gong. A multi-modal multi-objective evolutionary algorithm using two-archive and recombination strategies. IEEE Transactions on Evolutionary Computation, 2019, 23(4): 660-674.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'TriMOEATAR': {'integer', 'multi', 'multimodal', 'real'}}


class TriMOEATAR(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
