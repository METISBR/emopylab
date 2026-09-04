# emopylab 2026
"""Inverse modeling constrained MOEA/D.\n\nReference:\nL. R. C. Farias and A. F. R. Araujo. An inverse modeling constrained multi-objective evolutionary algorithm based on decomposition. Proceedings of the IEEE International Conference on Systems, Mans and Cybernetics, 2024.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'IMCMOEAD': {'constrained', 'integer', 'large', 'multi', 'real'}}


class IMCMOEAD(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
