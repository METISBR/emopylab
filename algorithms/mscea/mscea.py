# emopylab 2026
"""Multi-stage constrained multi-objective evolutionary algorithm.\n\nReference:\nY. Zhang, Y. Tian, H. Jiang, X. Zhang, and Y. Jin. Design and analysis of helper-problem-assisted evolutionary algorithm for constrained multiobjective optimization. Information Sciences, 2023, 648: 119547.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MSCEA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MSCEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
