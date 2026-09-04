# emopylab 2026
"""Evolutionary multitasking-based constrained multiobjective optimization.\n\nReference:\nK. Qiao, K. Yu, B. Qu, J. Liang, H. Song, and C. Yue. An evolutionary multitasking optimization framework for constrained multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2022, 26(2): 263-277.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'EMCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class EMCMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
