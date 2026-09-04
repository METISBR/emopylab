# emopylab 2026
"""Evolutionary multi-objective optimization with sparsity knowledge transfer.\n\nReference:\nC. Wu, Y. Tian, L. Zhang, X. Xiang, and X. Zhang. A sparsity knowledge transfer-based evolutionary algorithm for large-scale multitasking multi- objective optimization. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2582-2595.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'EMOSKT': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class EMOSKT(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
