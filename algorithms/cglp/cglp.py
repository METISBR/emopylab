# emopylab 2026
"""Correlation-guided layered prediction.

Reference:
K. Yu, D. Zhang, J. Liang, K. Chen, C. Yue, K. Qiao, and L. Wang. A correlation-guided layered prediction approach for evolutionary dynamic multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2025, 27(5): 1398-1412.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'CGLP': {'binary', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


class CGLP(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
