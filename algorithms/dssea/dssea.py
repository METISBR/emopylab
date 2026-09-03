# emopylab 2026
"""Dynamic subspace search-based evolutionary algorithm.\n\nReference:\nX. Ban, J. Liang, K. Yu, B. Qu, K. Qiao, P. N. Suganthan, and Y. Wang. A subspace search-based evolutionary algorithm for large-scale constrained multi-objective optimization and application. IEEE Transactions on Cybernetics, 2025, 55(5): 2486-2499.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'DSSEA': {'constrained', 'integer', 'large', 'many', 'multi', 'real'}}


class DSSEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
