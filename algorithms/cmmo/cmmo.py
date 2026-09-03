# emopylab 2026
"""Coevolutionary multi-modal multi-objective optimization framework.

Reference:
F. Ming, W. Gong, L. Wang, and L. Gao. Balancing convergence and diversity in objective and decision spaces for multimodal multi-objective optimization. IEEE Transactions on Emerging Topics in Computational Intelligence, 2023, 7(2): 474-486.
"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CMMO': {'binary', 'integer', 'label', 'multi', 'multimodal', 'permutation', 'real'}}


class CMMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type
