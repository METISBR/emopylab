# emopylab 2026
"""EMCMO with deep reinforcement learning-assisted operator selection.\n\nReference:\nF. Ming, W. Gong, L. Wang, and Y. Jin. Constrained multi-objective optimization with deep reinforcement learning assisted operator selection. IEEE/CAA Journal of Automatica Sinica, 2024, 11(4): 919-931.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DRLOSEMCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class DRLOSEMCMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
