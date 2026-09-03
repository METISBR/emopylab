# emopylab 2026
"""Enhanced competitive swarm optimizer for sparse optimization.\n\nReference:\nX. Wang, K. Zhang, J. Wang, and Y. Jin. An enhanced competitive swarm optimizer with strongly convex sparse operator for large-scale multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2022, 26(5): 859-871.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SECSO': {'large', 'multi', 'real', 'sparse'}}


class SECSO(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
