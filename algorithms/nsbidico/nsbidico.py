# emopylab 2026
"""Non-dominated sorting bidirectional differential coevolution algorithm.\n\nReference:\nC. S. R. Mendes, A. F. R. Araujo, and L. R. C. Farias. Non-dominated sorting bidirectional differential coevolution. Proceedings of the IEEE International Conference on Systems, Mans and Cybernetics, 2023.\n"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'NSBiDiCo': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class NSBiDiCo(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
