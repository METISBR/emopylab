# emopylab 2026
"""Dominance-weighted uniformity multi-objective evolutionary algorithm.\n\nReference:\nG. Moreira and L. Paquete. Guiding under uniformity measure in the decision space. Proceedings of the IEEE Latin American Conference on Computational Intelligence, 2019.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'DWU': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class DWU(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
