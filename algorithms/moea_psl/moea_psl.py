# emopylab 2026
"""Multi-objective evolutionary algorithm based on Pareto optimal subspace.\n\nReference:\nY. Tian, C. Lu, X. Zhang, K. C. Tan, and Y. Jin. Solving large-scale multi-objective optimization problems with sparse optimal solutions via unsupervised neural networks. IEEE Transactions on Cybernetics, 2021, 51(6): 3115-3128.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MOEAPSL': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class MOEAPSL(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
