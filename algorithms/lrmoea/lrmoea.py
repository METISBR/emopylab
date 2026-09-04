# emopylab 2026
"""Large-scale robust multi-objective evolutionary algorithm.\n\nReference:\nS. Shao, Y. Tian, L. Zhang, K. C. Tan, and X. Zhang. An evolutionary algorithm for solving large-scale robust multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2476-2490.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'LRMOEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'robust', 'sparse'}}


class LRMOEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
