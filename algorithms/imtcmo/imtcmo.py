# emopylab 2026
"""Improved evolutionary multitasking-based CMOEA.\n\nReference:\nK. Qiao, J. Liang, K. Yu, C. Yue, H. Lin, D. Zhang, and B. Qu. Evolutionary constrained multiobjective optimization: scalable high-dimensional constraint benchmarks and algorithm. IEEE Transactions on Evolutionary Computation, 2024, 28(4): 965-979.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'IMTCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class IMTCMO(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
