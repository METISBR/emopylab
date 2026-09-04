# emopylab 2026
"""Multi-stage constrained multi-objective evolutionary algorithm.\n\nReference:\nH. Ma, H. Wei, Y. Tian, R. Cheng, and X. Zhang. A multi-stage evolutionary algorithm for multi-objective optimization with complex constraints. Information Sciences, 2021, 560: 68-91.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MSCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MSCMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
