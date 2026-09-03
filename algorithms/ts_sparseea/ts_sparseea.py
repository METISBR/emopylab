# emopylab 2026
"""Two-stage SparseEA.\n\nReference:\nJ. Jiang, F. Han, J. Wang, Q. Ling, H. Han, and Y. Wang. A two-stage evolutionary algorithm for large-scale sparse multiobjective optimization problems. Swarm and Evolutionary Computation, 2022, 72: 101093.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'TSSparseEA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class TSSparseEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
