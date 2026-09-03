# emopylab 2026
"""Fletcher-Reeves conjugate gradient (for multi-objective optimization).\n\nReference:\nR. Fletcher and C. M. Reeves. Function minimization by conjugate gradients. The Computer Journal, 1964, 7(2): 149-154.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'FRCGM': {'constrained', 'large', 'many', 'multi', 'real'}}


class FRCGM(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
