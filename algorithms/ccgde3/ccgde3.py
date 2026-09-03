# emopylab 2026
"""CCGDE3 (Cooperative coevolution generalized differential evolution 3).

Reference:
L. M. Antonio and C. A. Coello Coello. Use of cooperative coevolution for solving large scale multiobjective optimization problems. CEC 2013, 2758-2765.
"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'CCGDE3': {'integer', 'large', 'multi', 'real'}}


class CCGDE3(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, num_groups=2, **kwargs):
        super().__init__(pop_size=pop_size, **kwargs)
        self.num_groups = num_groups
