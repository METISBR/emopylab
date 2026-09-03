# emopylab 2026
"""Interactive simple indicator-based evolutionary algorithm.\n\nReference:\nT. Chugh, K. Sindhya, J. Hakanen, and K. Miettinen. An interactive simple indicator-based evolutionary algorithm (I-SIBEA) for multiobjective optimization problems. Proceedings of the International Conference on Evolutionary Multi-Criterion Optimization, 2015, 277-291.\n"""

from __future__ import annotations

from algorithms.e_moea.e_moea import eMOEA


ALGORITHM_FLAGS = {'ISIBEA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class ISIBEA(eMOEA):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
