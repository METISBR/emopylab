# emopylab 2026
"""Constraints separation based evolutionary multitasking.\n\nReference:\nK. Qiao, J. Liang, K. Yu, X. Ban, C. Yue, B. Qu, and P. N. Suganthan. Constraints separation based evolutionary multitasking for constrained multi-objective optimization problems. IEEE/CAA Journal of Automatica Sinica, 2024, 11(8): 1819-1835.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'CSEMT': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CSEMT(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
