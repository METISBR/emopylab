# emopylab 2026
"""Utilizing the relationship between constrained and unconstrained Pareto fronts for constrained multi-objective optimization.\n\nReference:\nJ. Liang, K. Qiao, K. Yu, B. Qu, C. Yue, W. Guo, and L. Wang. Utilizing the relationship between unconstrained and constrained Pareto fronts for constrained multi-objective optimization. IEEE Transactions on Cybernetics, 2023, 53(6): 3873-3886.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'URCMO': {'constrained', 'integer', 'multi', 'real'}}


class URCMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
