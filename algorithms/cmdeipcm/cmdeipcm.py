# emopylab 2026
"""Constrained multiobjective differential evolution algorithm with an infeasible proportion control mechanism.

Reference:
J. Liang, X. Ban, K. Yu, K. Qiao, and B. Qu. Constrained multiobjective differential evolution algorithm with infeasible-proportion control mechanism. Knowledge-Based Systems, 2022, 250: 109105.
"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'CMDEIPCM': {'constrained', 'integer', 'large', 'multi', 'real'}}


class CMDEIPCM(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
