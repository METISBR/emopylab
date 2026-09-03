# emopylab 2026
"""Two-phase framework with NSGA-II.\n\nReference:\nZ. Liu and Y. Wang. Handling constrained multiobjective optimization problems with constraints in both the decision and objective spaces. IEEE Transactions on Evolutionary Computation, 2019, 23(5): 870-884.\n"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'ToP': {'constrained', 'integer', 'multi', 'real'}}


class ToP(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
