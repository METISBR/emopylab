# emopylab 2026
"""Decision variables classification-based evolutionary algorithm.\n\nReference:\nX. Ban, J. Liang, K. Qiao, K. Yu, Y. Wang, J. Zhu, B. Qu. A decision variables classification-based evolutionary algorithm for constrained multi-objective optimization problems. IEEE/CAA Journal of Automatica Sinica, 2025, 12(9): 1830-1849.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'DVCEA': {'constrained', 'integer', 'large', 'many', 'multi', 'real'}}


class DVCEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
