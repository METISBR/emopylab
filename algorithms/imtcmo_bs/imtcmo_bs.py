# emopylab 2026
"""Improved evolutionary multitasking-based CMOEA with bidirectional sampling.\n\nReference:\nK. Qiao, J. Liang, K. Yu, W. Guo, C. Yue, B. Qu, and P. N. Suganthan. Benchmark problems for large-scale constrained multi-objective optimization with baseline results. Swarm and Evolutionary Computation, 2024, 86: 101504.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'IMTCMO_BS': {'binary', 'constrained', 'integer', 'label', 'large', 'many', 'multi', 'permutation', 'real'}}


class IMTCMO_BS(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
