# emopylab 2026
"""Dual-population with dynamic constraint processing and resource allocating.\n\nReference:\nK. Qiao, Z. Chen, B. Qu, K. Yu, C. Yue, K. Chen, and J. Liang. A dual- population evolutionary algorithm based on dynamic constraint processing and resources allocation for constrained multi-objective optimization problems. Expert Systems With Applications, 2024, 238: 121707.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DPCPRA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class DPCPRA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
