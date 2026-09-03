# emopylab 2026
"""Large-scale constrained multi-objective evolutionary algorithm.\n\nReference:\nL. Si, X. Zhang, Y. Zhang, S. Yang, and Y. Tian. An efficient sampling approach to offspring generation for evolutionary large-scale constrained multi-objective optimization. IEEE Transactions on Emerging Topics in Computational Intelligence, 2025, 9(3): 2080-2092.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'LCMEA': {'constrained', 'large', 'multi', 'real'}}


class LCMEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
