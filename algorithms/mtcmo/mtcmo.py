# emopylab 2026
"""Multitasking constrained multi-objective optimization.\n\nReference:\nK. Qiao, K. Yu, B. Qu, J. Liang, H. Song, C. Yue, H. Lin, and K. C. Tan. Dynamic auxiliary task-based evolutionary multitasking for constrained multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2023, 27(3): 642-656.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MTCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MTCMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
