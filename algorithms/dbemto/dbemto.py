# emopylab 2026
"""Double-balanced evolutionary multi-task optimization.\n\nReference:\nK. Qiao, J. Liang, K. Yu, M. Wang, B. Qu, C. Yue, and Y. Gou. A self-adaptive evolutionary multi-task based constrained multi-objective evolutionary algorithm. IEEE Transactions on Emerging Topics in Computational Intelligence, 2023, 7(4): 1098-1112.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DBEMTO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class DBEMTO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
