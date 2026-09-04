# emopylab 2026
"""Two-stage evolutionary algorithm with three indicators.\n\nReference:\nJ. Dong, W. Gong, F. Ming, and L. Wang. A two-stage evolutionary algorithm based on three indicators for constrained multi-objective optimization. Expert Systems with Applications, 2022, 195: 116499.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'TSTI': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class TSTI(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
