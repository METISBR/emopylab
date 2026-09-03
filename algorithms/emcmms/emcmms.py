# emopylab 2026
"""Evolutionary multitasking with a cooperative multistep mutation strategy.\n\nReference:\nK. Qiao, K. Yu, C. Yue, B. Qu, M. Liu, and J. Liang. A cooperative multistep mutation strategy for multiobjective optimization problems with deceptive constraints. IEEE Transactions on Systems, Man, and Cybernetics, 2024, 54(11): 6670-6682.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'EMCMMS': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class EMCMMS(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
