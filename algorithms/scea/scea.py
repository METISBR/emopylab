# emopylab 2026
"""Sparsity clustering basec evolutionary algorithm.\n\nReference:\nY. Zhang, C. Wu, Y. Tian, and X. Zhang. A co-evolutionary algorithm based on sparsity clustering for sparse large-scale multi-objective optimization. Engineering Applications of Artificial Intelligence, 2024, 133: 108194\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'SCEA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class SCEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
