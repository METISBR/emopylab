# emopylab 2026
"""Multiobjective multitask evolutionary algorithm based on decomposition with dual neighborhoods.\n\nReference:\nX. Wang, Z. Dong, L. Tang, and Q. Zhang. Multiobjective multitask optimization-neighborhood as a bridge for knowledge transfer. IEEE Transactions on Evolutionary Computation, 2023, 27(1): 155-169.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MTEADDN': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MTEADDN(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
