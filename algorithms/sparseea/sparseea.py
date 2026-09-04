# emopylab 2026
"""Evolutionary algorithm for sparse multi-objective optimization problems.\n\nReference:\nY. Tian, X. Zhang, C. Wang, and Y. Jin. An evolutionary algorithm for large-scale sparse multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2020, 24(2): 380-393.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'SparseEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class SparseEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
