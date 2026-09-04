# emopylab 2026
"""Constrained multi-objective evolutionary algorithm with self-organizing map.\n\nReference:\nC. He, M. Li, C. Zhang, H. Chen, P. Zhong, Z. Li, and J. Li. A self-organizing map approach for constrained multi-objective optimization problems. Complex & Intelligent Systems, 2022, 8: 5355-5375.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CMOSMA': {'constrained', 'integer', 'many', 'multi', 'real'}}


class CMOSMA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
