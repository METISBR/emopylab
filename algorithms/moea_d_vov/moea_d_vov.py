# emopylab 2026
"""MOEA/D with virtual objective vectors.\n\nReference:\nT. Takagi, K. Takadama, and H. Sato. Weight vector arrangement using virtual objective vectors in decomposition-based MOEA. Proceedings of the IEEE Congress on Evolutionary Computation, 2021, 1462-1469.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADVOV': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MOEADVOV(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
