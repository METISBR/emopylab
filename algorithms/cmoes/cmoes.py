# emopylab 2026
"""Constrained multi-objective optimization based on even search.\n\nReference:\nF. Ming, W. Gong, and Y. Jin. Even search in a promising region for constrained multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2024, 11(2): 474-486.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CMOES': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CMOES(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
