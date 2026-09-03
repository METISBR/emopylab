# emopylab 2026
"""Preference-inspired coevolutionary algorithm with goals.\n\nReference:\nR. Wang, R. C. Purshouse, and P. J. Fleming. Preference-inspired coevolutionary algorithms for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2013, 17(4): 474-494.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'PICEAg': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class PICEAg(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
