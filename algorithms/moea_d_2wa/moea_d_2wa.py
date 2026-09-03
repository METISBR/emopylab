# emopylab 2026
"""MOEA/D with two-type weight vector adjustments.\n\nReference:\nR. Jiao, S. Zeng, C. Li, and Y. S. Ong. Two-type weight adjustments in MOEA/D for highly constrained many-objective optimization. Information Sciences, 2021, 578: 592-614.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MOEAD2WA': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MOEAD2WA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
