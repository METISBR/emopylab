# emopylab 2026
"""Non-dominated sorting differential evolution with prediction in the objective space.\n\nReference:\nE. Guerrero-Pena and A. F. R. Araujo. Multi-objective evolutionary algorithm with prediction in the objective space. Information Sciences, 2019, 501: 293-316.\n"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'OSPNSDE': {'integer', 'multi', 'real'}}


class OSPNSDE(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
