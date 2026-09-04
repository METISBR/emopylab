# emopylab 2026
"""Self-organized surrogate-assisted differential evolution.\n\nReference:\nA. F. R. Araújo, L. R. C. Farias, and A. R. C. Gonçalves. Self-organizing surrogate-assisted non-dominated sorting differential evolution. Swarm and Evolutionary Computation, 2024, 91: 101703.\n"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'SSDE': {'constrained', 'expensive', 'integer', 'many', 'multi', 'real'}}


class SSDE(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
