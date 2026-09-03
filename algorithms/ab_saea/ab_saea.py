# emopylab 2026
"""AB-SAEA (Adaptive Bayesian surrogate-assisted EA).

Reference:
X. Wang, Y. Jin, S. Schmitt, and M. Olhofer. An adaptive Bayesian approach to surrogate-assisted evolutionary multi-objective optimization. Information Sciences, 2020, 519: 317-331.
"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'ABSAEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


class ABSAEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, alpha=2, wmax=20, mu=5, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha = alpha
        self.wmax = wmax
        self.mu = mu
