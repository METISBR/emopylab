# emopylab 2026
"""Ensemble-based surrogate model-assisted evolutionary algorithm.\n\nReference:\nY. Li, X. Feng, and H. Yu. Enhancing landscape approximation with ensemble-based surrogate model for expensive constrained multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2025.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'EM_SAEA': {'constrained', 'expensive', 'many', 'multi', 'real'}}


class EM_SAEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
